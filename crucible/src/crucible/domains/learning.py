"""Learner knowledge, educational artifacts and independent assessment (kernel module `learn`, host `learn`).

The world of the education ecosystem (MASTER_SPEC 9.10). Its parts:

* **Curriculum.** A set of executable concepts (`domains.concepts`) closed under prerequisites, published by the
  Board of Education as a public document and through `learn.curriculum`.
* **Learner knowledge state.** Each learner (an NPC principal with its own account) holds, per concept and level
  (1-3), a mastery value; per concept a memory half-life, the time of its last study and a misconception strength.
  The state is private to the learner (events visible to the learner only) and read by the evaluator.
* **Educational artifacts.** Any public document. A learner reads only lines `Example: <problem> => <answer>` (a
  worked example) and `Practice: <problem> => <answer>` (an exercise it attempts before reading the answer), plus
  an optional `Next: <doc id>` link; problems are recognised by the curriculum's formats and every stated answer
  is checked by the world's solver. Prose is not read: time spent on a document teaches nothing by itself.
* **Learning dynamics** (documented in FIDELITY_REGISTER.md): gains scale with readiness (prerequisite and lower-level
  mastery), novelty (an item already seen teaches little), massing (repeats of one concept-level in one session
  teach less) and expertise (worked examples help novices most, practice helps learners who already know some);
  practice with feedback lengthens the half-life, more when spaced; wrong answer keys and wrong examples create or
  strengthen misconceptions; reading costs attention minutes and a session ends when the learner's minutes run out.
* **Retention.** Effective mastery decays between studies towards a floor with the concept's half-life:
  `m_eff = m * (floor + (1 - floor) * 2 ** (-days_since_study / half_life))`.
* **Participation.** Learners decide for themselves: whether to study (motivation), what (their goal, or the weakest
  prerequisite they perceive), where to look (search or forum recommendations, the calibrated slate-click model,
  their own trust in authors learned from their own experience), whether to follow a lesson's `Next:` link, when to
  give up (dropout) and come back, and whether to recommend a lesson after an assessment showed them progress.
* **Independent assessment.** A placement test when a learner joins and a periodic test every `assess_days`, on
  problems generated fresh for each test and never present in any published document. Answers come from the
  learner's knowledge state (`answer`), are graded by the solver, and only the learner sees its own scores; public
  aggregates are published.
* **Teachers.** NPC teachers with different styles publish lessons and respond to the public scores.

Handlers read state and emit events; appliers write. Randomness comes from `ctx.rng` / `world.rng_for`.
"""
from __future__ import annotations

import math
import re

from ..core import schema as S
from ..core.canonical import digest
from ..core.errors import CrucibleError, Forbidden, InvalidAction, NotFound
from ..core.kernel import Module, applier, endpoint, process, register_module
from . import concepts as C
from .common import DAY, HOUR, SlateClicks, bow_cos, bump_hourly, ctr_features, make_person, make_seat

DEFAULTS = {
    "curriculum": ["add", "sub", "mul", "div", "gcd", "lcm", "frac_simplify", "frac_add"],
    "n": 40, "prefix": "l", "background": [2.0, 2.0], "minutes": [15, 45], "rhythm_h": [12, 36],
    "motivation": [0.55, 0.9], "channels": {"search": 1.0}, "half_life_days": [3.0, 8.0], "floor": 0.25,
    "assess_days": 7, "assess_k": 3, "placement_k": 8, "tick_h": 6, "board": "study", "recommend_p": 0.5,
    "dropout": 0.12, "return_p_day": 0.03, "teachers": [], "peer_p": 0.0, "cohorts": [], "seat": None,
    "trust_lr": 0.6, "goal_min_depth": 2,
}
# learning-rate constants of the dynamics (FIDELITY_REGISTER.md rows "education.*")
A_EX, A_PR, THETA, GUESS, SLIP, B_WRONG, H_MAX = 0.30, 0.34, 0.6, 0.02, 0.05, 0.22, 120.0
MIN_EX, MIN_PR = 1.5, 2.5          # attention minutes per item and level
NOVEL_REPEAT, MASS = 0.25, 0.35
LEVELS = (1, 2, 3)
DOC_RE = re.compile(r"doc_[0-9a-z_]+")


def _u(rng, lohi, integer=False):
    lo, hi = lohi if isinstance(lohi, (list, tuple)) else (lohi, lohi)
    return int(rng.integers(int(lo), int(hi) + 1)) if integer else float(rng.uniform(lo, hi))


def _cl(x, lo=0.0, hi=1.0):
    return lo if x < lo else (hi if x > hi else x)


# ------------------------------------------------------------------------------------------ the learner model
def eff(rec, c: str, L: int, t: int, floor: float) -> float:
    """Effective mastery of (concept, level) at time t: the stored mastery decayed by the forgetting curve."""
    m = rec["m"][c][L - 1]
    dt = max(0.0, (t - rec["tl"][c]) / DAY)
    return m * (floor + (1.0 - floor) * 2.0 ** (-dt / max(1e-6, rec["h"][c])))


def readiness(rec, c: str, L: int, t: int, floor: float, cur) -> float:
    pre = [p for p in C.PREREQS[c] if p in cur]
    r = min((_cl(eff(rec, p, L, t, floor) / THETA) for p in pre), default=1.0)
    if L > 1:
        r *= _cl(eff(rec, c, L - 1, t, floor) / THETA)
    return r


def p_correct(rec, c: str, L: int, t: int, floor: float, cur) -> float:
    pre = [p for p in C.PREREQS[c] if p in cur]
    pf = min((0.4 + 0.6 * eff(rec, p, L, t, floor) for p in pre), default=1.0)
    return GUESS + (1.0 - GUESS - SLIP) * eff(rec, c, L, t, floor) * pf


def answer(rec, c: str, p, t: int, floor: float, cur, rng) -> tuple[str, bool]:
    """The learner's answer to problem `p` of concept `c` (a misconception, a slip or the solution)."""
    con = C.LIBRARY[c]
    L = int(con.level(p))
    me = eff(rec, c, L, t, floor)
    if rng.random() < rec["bug"][c] * (1.0 - 0.5 * me):
        a = con.buggy(p)
    elif rng.random() < p_correct(rec, c, L, t, floor, cur):
        a = con.solve(p)
    else:
        a = _slip(con, p, rng)
    return a, con.grade(a, p)


def _slip(con, p, rng) -> str:
    sol = con.solve(p)
    v = con.norm(sol)
    if isinstance(v, int):
        return str(v + int(rng.choice([-10, -2, -1, 1, 2, 10])))
    if isinstance(v, float):
        return str(round(v + float(rng.choice([-1.0, -0.5, 0.5, 1.0])), 2))
    if isinstance(v, tuple):
        return f"{v[0]} r {v[1] + 1}"
    if isinstance(v, str):
        return v + "1"
    from fractions import Fraction
    if isinstance(v, Fraction):
        return f"{v.numerator + 1}/{v.denominator}"
    return "?"


def assess(rec, t: int, rng, k: int, cur, floor: float, exclude: set) -> tuple[dict, int, str]:
    """Scores on fresh problems (k per concept and level, none of them in `exclude`)."""
    scores, keys = {}, []
    for c in cur:
        ok = 0
        for L in LEVELS:
            for _ in range(k):
                p = C.generate(c, rng, L)
                for _ in range(10):
                    if C.LIBRARY[c].key(p) not in exclude:
                        break
                    p = C.generate(c, rng, L)
                keys.append(C.LIBRARY[c].key(p))
                ok += answer(rec, c, p, t, floor, cur, rng)[1]
        scores[c] = round(ok / (3 * k), 4)
    return scores, len(keys), digest(keys)


def published_problem_keys(world, cur) -> set:
    """Every problem that appears in a lesson line of any public document (assessments never reuse them)."""
    st = world.state
    out = set()
    if "documents" not in st.tables:
        return out
    for did in st.find_ids("documents", "visibility", "public"):
        for it in C.parse_lesson(st.tables["documents"][did]["body"], cur, max_lines=200)["items"]:
            out.add(it["key"])
    return out


@register_module
class Learning(Module):
    name = "learn"
    host = "learn"
    depends = ("identity", "bank", "docs", "search", "forum")

    def install(self, world):
        st = world.state
        st.declare_table("learn_learners", index=("state",))
        st.declare_table("learn_sessions", index=("learner", "doc", "author"))
        st.declare_table("learn_assessments", index=("learner", "kind"))
        st.declare_table("learn_docstats", index=("author",))
        st.declare_table("learn_teachers", index=())
        self.cfg = {**DEFAULTS, **self.config}
        self.cur = C.closure(self.cfg["curriculum"])

    # ------------------------------------------------------------------ genesis
    def genesis(self, ctx):
        cfg = self.cfg
        rng = ctx.world.rng_for("learn-genesis")
        if cfg.get("seat"):
            make_seat(ctx, cfg["seat"]["handle"], int(cfg["seat"].get("endowment", 0)))
        from .common import make_org
        board, lead = make_org(ctx, "board-of-education", "Board of Education", "institution")
        docs = ctx.world.modules["docs"]
        docs.create_doc(ctx, board, "Curriculum: concepts, prerequisites and problem formats", self.curriculum_text(),
                        "public")
        ctx.emit("learn.meta", {"board": board}, vis="public")
        groups = [(0, int(cfg["n"]))] + [(int(c["at_day"]) * DAY, int(c["n"])) for c in cfg["cohorts"]]
        i = 0
        chans = sorted(cfg["channels"])
        cw = [float(cfg["channels"][k]) for k in chans]
        for joined, n in groups:
            for _ in range(n):
                lid = make_person(ctx, f"{cfg['prefix']}{i:03d}", f"Learner {i}")
                ctx.emit("learn.learner", self._new_learner(rng, i, lid, joined, chans, cw), vis="internal")
                i += 1
        for j, t in enumerate(cfg["teachers"]):
            pid = make_person(ctx, t["handle"], t.get("name", t["handle"]), kind="npc", role="teacher")
            ctx.emit("learn.teacher", {"id": pid, "style": t["style"], "per_week": float(t.get("per_week", 2)),
                                       "name": t.get("name", t["handle"]), "covered": {}}, vis="internal")
        ctx.schedule(ctx.now + 60, "learn.tick", {}, prio=60, period=int(cfg["tick_h"]) * HOUR, key="learn.tick")
        ctx.schedule(ctx.now + int(cfg["assess_days"]) * DAY, "learn.assess", {}, prio=55,
                     period=int(cfg["assess_days"]) * DAY, key="learn.assess")
        if cfg["teachers"]:
            ctx.schedule(ctx.now + 30, "learn.teach", {}, prio=58, period=DAY, key="learn.teach")

    def _new_learner(self, rng, i, lid, joined, chans, cw):
        cfg = self.cfg
        cur = self.cur
        b = float(rng.beta(*cfg["background"]))
        m, h, tl, bug = {}, {}, {}, {}
        for c in cur:
            d = C.depth(c, cur)
            base = _cl(b * (1.25 - 0.28 * d) + float(rng.normal(0, 0.08)), 0.0, 0.97)
            m[c] = [round(base, 4), round(base * 0.7, 4), round(base * 0.4, 4)]
            h[c] = round(_u(rng, cfg["half_life_days"]) * (1.0 + 3.0 * base), 3)
            tl[c] = int(joined)
            bug[c] = 0.0
        deep = [c for c in cur if C.depth(c, cur) >= int(cfg["goal_min_depth"]) and m[c][0] < 0.6] or \
            sorted(cur, key=lambda c: -C.depth(c, cur))[:1]
        w = [1.0 + C.depth(c, cur) for c in deep]
        goal = deep[int(rng.choice(len(deep), p=[x / sum(w) for x in w]))]
        return {"id": lid, "i": i, "goal": goal, "state": "waiting", "joined_at": int(joined),
                "motivation": round(_u(rng, cfg["motivation"]), 4), "m": m, "h": h, "tl": tl, "bug": bug,
                "perceived": {}, "trust": {}, "seen": [], "minutes": _u(rng, cfg["minutes"], True),
                "rhythm_h": _u(rng, cfg["rhythm_h"], True),
                "channel": chans[int(rng.choice(len(chans), p=[x / sum(cw) for x in cw]))],
                "next_session": int(joined), "sessions": 0, "scores": {}, "placement": {}, "next_doc": None,
                "avoid": [], "hist_c": {}, "hist_a": {}, "posted": [], "notes": []}

    def curriculum_text(self) -> str:
        lines = ["Concepts taught and assessed in this world, with prerequisites and problem formats.", ""]
        for c in self.cur:
            con = C.LIBRARY[c]
            pre = ", ".join(p for p in C.PREREQS[c] if p in self.cur) or "none"
            lines.append(f"- {c}: {con.title}. {con.doc} Problems are written `{con.fmt}`; answers: {con.answer_fmt}. "
                         f"Prerequisites: {pre}.")
        lines += ["", "How learners read lessons: a line `Example: <problem> => <answer>` is a worked example; a line "
                  "`Practice: <problem> => <answer>` is an exercise the learner attempts before reading the answer; a "
                  "line `Next: <doc id>` points to the next lesson. Problems must use the formats above. Other text is "
                  "not read. Learners are assessed on new problems of every concept and level."]
        return "\n".join(lines)

    # ------------------------------------------------------------------ helpers
    def floor(self) -> float:
        return float(self.cfg["floor"])

    def learner(self, st, lid):
        r = st.get("learn_learners", lid)
        if r is None:
            raise Forbidden("only enrolled learners study (see learn.curriculum)")
        return r

    def _call(self, ctx, p, tool, args):
        try:
            return True, ctx.call(tool, args, as_principal=p)
        except CrucibleError as e:
            return False, e.as_dict()

    def _change(self, ctx, lid, **changes):
        ctx.emit("learn.state", {"id": lid, "changes": changes}, vis=(lid,))

    # ------------------------------------------------------------------ public tools
    @endpoint("curriculum", S.obj(), read_only=True)
    def ep_curriculum(self, ctx):
        """The concepts of this world: name, title, problem and answer formats, prerequisites, and how learners read
        lessons."""
        return {"concepts": [{"name": c, "title": C.LIBRARY[c].title, "doc": C.LIBRARY[c].doc,
                              "format": C.LIBRARY[c].fmt, "answer_format": C.LIBRARY[c].answer_fmt,
                              "prerequisites": [p for p in C.PREREQS[c] if p in self.cur]} for c in self.cur],
                "how_learners_read": self.curriculum_text().split("How learners read lessons: ")[1]}

    @endpoint("check", S.obj({"body": S.s_str("lesson text to check", maxLength=100_000),
                              "doc": S.s_str("or a document you can read", maxLength=40)}), read_only=True)
    def ep_check(self, ctx, body=None, doc=None):
        """Which lines of a lesson learners will read (worked examples and exercises with their concept and level),
        and its Next link. Says nothing about whether the stated answers are right."""
        if doc:
            d, _ = ctx.world.modules["docs"]._doc(ctx, doc, 1)
            body = d["body"]
        if body is None:
            raise InvalidAction("give body or doc")
        les = C.parse_lesson(body, self.cur)
        return {"items": [{"kind": i["kind"], "concept": i["concept"], "level": i["level"]} for i in les["items"]],
                "next": les["next"], "minutes_at_level": sum((MIN_EX if i["kind"] == "example" else MIN_PR) * i["level"]
                                                             for i in les["items"])}

    @endpoint("stats", S.obj(), read_only=True)
    def ep_stats(self, ctx):
        """Public education statistics: the latest assessment's mean score per concept, active learners, sessions in
        the last 7 days, and the most-read lessons (readers, sessions)."""
        st = ctx.state
        summ = st.meta_get("learn:summary", {})
        week = ctx.now - 7 * DAY
        recent = sum(1 for _, s in st.scan("learn_sessions") if s["t"] >= week)
        docs = sorted(st.scan("learn_docstats"), key=lambda kv: (-kv[1]["n_readers"], kv[0]))[:20]
        return {"assessment": summ, "active_learners": len(st.find_ids("learn_learners", "state", "active")),
                "sessions_7d": recent,
                "lessons": [{"doc": k, "author": v["author"], "readers": v["n_readers"], "sessions": v["sessions"]}
                            for k, v in docs]}

    @endpoint("readers", S.obj(), read_only=True)
    def ep_readers(self, ctx):
        """For authors: how many distinct learners studied each of your documents and how many sessions there were."""
        return [{"doc": k, "readers": v["n_readers"], "sessions": v["sessions"]}
                for k, v in sorted(ctx.state.scan("learn_docstats")) if v["author"] == ctx.actor]

    @endpoint("me", S.obj(), read_only=True)
    def ep_me(self, ctx):
        """Learners only: your goal, your own test scores and how you rate your own progress."""
        r = ctx.state.get("learn_learners", ctx.actor)
        if r is None:
            return {"learner": False}
        return {"learner": True, "goal": r["goal"], "state": r["state"], "scores": dict(r["scores"]),
                "placement": dict(r["placement"]), "perceived": dict(r["perceived"]), "sessions": r["sessions"]}

    @endpoint("study", S.obj({"doc": S.s_str(maxLength=40)}, ["doc"]), cost_s=600)
    def ep_study(self, ctx, doc):
        """Learners only: study a document you can read (its worked examples and exercises, in order, until your
        attention for this session runs out)."""
        st = ctx.state
        r = self.learner(st, ctx.actor)
        if r["state"] != "active":
            raise Forbidden("you are not studying at the moment")
        d, _ = ctx.world.modules["docs"]._doc(ctx, doc, 1)
        les = C.parse_lesson(d["body"], self.cur)
        return self._study(ctx, r, doc, d, les)

    def _study(self, ctx, r, doc, d, les):
        cfg, cur, fl, now = self.cfg, self.cur, self.floor(), ctx.now
        rng = ctx.rng("learn.study", r["id"], doc)
        m = {c: list(v) for c, v in r["m"].items()}
        h, tl, bug = dict(r["h"]), dict(r["tl"]), dict(r["bug"])
        work = {"m": m, "h": h, "tl": tl, "bug": bug}
        seen = set(r["seen"])
        touched: dict[str, float] = {}
        massed: dict[str, int] = {}
        minutes = float(r["minutes"])
        used = 0.0
        out_items, practice = [], []
        perceived_ok: dict[str, list] = {}
        for it in les["items"]:
            c, L = it["concept"], it["level"]
            cost = (MIN_EX if it["kind"] == "example" else MIN_PR) * L
            if used + cost > minutes:
                break
            used += cost
            if c not in touched:            # consolidate the forgetting curve before studying the concept again
                for lv in LEVELS:
                    m[c][lv - 1] = eff(work, c, lv, now, fl)
                touched[c] = min(1.0, max(0.0, (now - tl[c]) / DAY) / max(1e-6, h[c]))
                tl[c] = now
            spacing = touched[c]
            R = readiness(work, c, L, now, fl, cur)
            nov = 1.0 if it["key"] not in seen else NOVEL_REPEAT
            k = massed.get(f"{c}|{L}", 0)
            massed[f"{c}|{L}"] = k + 1
            mass = 1.0 / (1.0 + MASS * k)
            mm = m[c][L - 1]
            rec = {"kind": it["kind"], "concept": c, "level": L, "key_correct": it["correct"], "readiness": round(R, 3)}
            if it["kind"] == "example":
                if it["correct"]:
                    m[c][L - 1] = mm + A_EX * R * nov * mass * (1 - mm) * (1 - 0.7 * mm)
                    h[c] = min(H_MAX, h[c] * (1 + 0.15 * spacing * R))
                else:
                    bug[c] = bug[c] + B_WRONG * (1 - mm) * (1 - bug[c])
                    m[c][L - 1] = mm * 0.96
            else:
                p = tuple(it["params"])
                ans, truly = answer(work, c, p, now, fl, cur, rng)
                con = C.LIBRARY[c]
                matches_key = con.norm(ans) is not None and con.norm(ans) == con.norm(it["stated"])
                perceived_ok.setdefault(c, []).append(matches_key)
                practice.append({"problem": con.text(p), "your_answer": ans, "answer_key": it["stated"],
                                 "matched": matches_key})
                rec["attempt_correct"] = truly
                if it["correct"]:
                    m[c][L - 1] = mm + A_PR * R * nov * mass * (1 - mm) * (0.3 + 0.7 * mm) * (1.0 if truly else 0.6)
                    h[c] = min(H_MAX, h[c] * (1 + (0.6 if truly else 0.1) * spacing * R))
                    if not truly:
                        bug[c] = bug[c] * 0.8
                elif truly:              # a right attempt contradicted by a wrong key: confusion
                    bug[c] = bug[c] + 0.5 * B_WRONG * (1 - bug[c])
                    m[c][L - 1] = mm * 0.97
                else:                    # a wrong attempt confirmed by a wrong key
                    bug[c] = bug[c] + B_WRONG * (1 - bug[c])
            m[c][L - 1] = round(_cl(m[c][L - 1]), 5)
            bug[c] = round(_cl(bug[c]), 5)
            h[c] = round(h[c], 4)
            seen.add(it["key"])
            out_items.append(rec)
        usable = len(out_items)
        # the learner's own reading of the session: did its attempts match the answer key?
        perceived = dict(r["perceived"])
        for c, oks in perceived_ok.items():
            perceived[c] = round(0.7 * perceived.get(c, 0.5) + 0.3 * (sum(oks) / len(oks)), 4)
        all_ok = [x for v in perceived_ok.values() for x in v]
        success = sum(all_ok) / len(all_ok) if all_ok else (0.5 if usable else 0.0)
        author = d["owner"]
        trust = dict(r["trust"])
        old = trust.get(author, 1.0)
        trust[author] = round(_cl(old + float(cfg["trust_lr"]) * (success - 0.5) - (0.4 if not usable else 0.0),
                                  0.1, 2.5), 4)
        mot = _cl(r["motivation"] + (0.08 * (success - 0.4) if usable else -0.06) - 0.02)
        n_doc = sum(1 for s in ctx.state.find("learn_sessions", "learner", r["id"]) if s["doc"] == doc) + 1
        avoid = list(r["avoid"])
        if (not usable or n_doc >= 3) and doc not in avoid:
            avoid = (avoid + [doc])[-50:]
        hist_c = dict(r["hist_c"])
        for c in touched:
            hist_c[c] = hist_c.get(c, 0) + 1
        hist_a = dict(r["hist_a"])
        hist_a[author] = hist_a.get(author, 0) + 1
        changes = {"m": m, "h": h, "tl": tl, "bug": bug, "seen": sorted(seen)[-600:], "perceived": perceived,
                   "trust": trust, "motivation": round(mot, 4), "avoid": avoid, "hist_c": hist_c, "hist_a": hist_a,
                   "next_doc": les["next"] if (les["next"] and success >= 0.5 and les["next"] != doc) else None,
                   "sessions": r["sessions"] + 1}
        sid = ctx.state.new_id("lsess")
        ctx.emit("learn.studied", {"id": sid, "learner": r["id"], "doc": doc, "version": d["version"], "author": author,
                                   "items": out_items, "minutes": round(used, 2), "usable": usable,
                                   "concepts": sorted(touched), "changes": changes}, vis=(r["id"],))
        return {"session": sid, "items_read": usable, "minutes": round(used, 2), "practice": practice[:20],
                "next": les["next"]}

    # ------------------------------------------------------------------ the learners' own decisions
    def choose_concept(self, r) -> str:
        c = r["goal"]
        for _ in range(8):
            weak = [p for p in C.PREREQS[c] if p in self.cur and r["perceived"].get(p, 0.5) < 0.55]
            if not weak:
                return c
            c = min(weak, key=lambda p: (r["perceived"].get(p, 0.5), p))
        return c

    def _relevant(self, con, text: str) -> bool:
        t = text.lower()
        return con.name.lower() in t or con.title.lower() in t

    def discover(self, ctx, lid, r, c, rng) -> str | None:
        con = C.LIBRARY[c]
        st = ctx.state
        avoid = set(r["avoid"])
        if r["channel"] == "forum":
            ok, posts = self._call(ctx, lid, "forum.list", {"board": self.cfg["board"], "sort": "new", "limit": 20})
            if ok:
                for post in posts:
                    if not self._relevant(con, post["title"] + " " + post.get("snippet", "")):
                        continue
                    ok2, full = self._call(ctx, lid, "forum.read", {"post": post["post"], "replies": 0})
                    if not ok2:
                        continue
                    for did in DOC_RE.findall(full.get("body", "")):
                        if did not in avoid and st.has("documents", did):
                            return did
        q = f"{con.title} lesson {con.name}"
        ok, res = self._call(ctx, lid, "search.query", {"q": q, "kind": "doc", "limit": 8})
        if not ok:
            return None
        slate = [x for x in res["results"] if x.get("ref") and x["ref"] not in avoid and not x.get("sponsored")]
        if not slate:
            return None
        n_hist = sum(r["hist_c"].values())
        tot_a = sum(r["hist_a"].values())
        prior = SlateClicks.prior()
        active = max(1, len(st.find_ids("learn_learners", "state", "active")))
        cos, cat, sub, ctr, cpk, age, rel = [], [], [], [], [], [], []
        for x in slate:
            text = x["title"] + " " + x.get("snippet", "")
            cos.append(bow_cos(q, text))
            rc = next((k for k in self.cur if self._relevant(C.LIBRARY[k], x["title"])), None)
            share = 1.0 / len(self.cur)
            cat.append((r["hist_c"].get(rc, 0) + share) / (n_hist + 1) if rc else share / (n_hist + 1))
            sub.append((r["hist_a"].get(x["owner"], 0) + 0.01) / (tot_a + 1))
            s = st.get("learn_docstats", x["ref"])
            a, b = ctr_features(s["hourly"] if s else None, ctx.now, active, prior)
            ctr.append(a)
            cpk.append(b)
            age.append((ctx.now - (x.get("t") or ctx.now)) / HOUR)
            rel.append(self._relevant(con, text))
        p = SlateClicks.probabilities(SlateClicks.features(cos, cat, sub, n_hist, ctr, cpk, age))
        w = [float(p[i]) * r["trust"].get(x["owner"], 1.0) * (1.0 if rel[i] else 0.05) for i, x in enumerate(slate)]
        tot = sum(w)
        pick = int(rng.choice(len(slate), p=[v / tot for v in w])) if tot > 0 else 0
        ctx.emit("learn.browse", {"learner": lid, "shown": [x["ref"] for x in slate], "opened": slate[pick]["ref"],
                                  "query": q}, vis=(lid,))
        return slate[pick]["ref"]

    def step(self, ctx, lid, r, rng):
        cfg = self.cfg
        nxt = ctx.now + int(r["rhythm_h"] * HOUR * (0.75 + 0.5 * rng.random()))
        if rng.random() > r["motivation"]:
            self._change(ctx, lid, next_session=nxt)
            return
        goal = r["goal"]
        if r["perceived"].get(goal, 0.0) >= 0.85:
            nxt_goal = [c for c in self.cur if r["perceived"].get(c, 0.5) < 0.8 and c != goal and
                        all(r["perceived"].get(p, 0.5) >= 0.55 for p in C.PREREQS[c] if p in self.cur)]
            if not nxt_goal:
                ctx.emit("learn.decision", {"learner": lid, "kind": "done", "goal": goal}, vis=(lid,))
                self._change(ctx, lid, state="done", next_session=nxt)
                return
            goal = sorted(nxt_goal, key=lambda c: (-C.depth(c, self.cur), c))[0]
            ctx.emit("learn.decision", {"learner": lid, "kind": "goal", "goal": goal}, vis=(lid,))
            self._change(ctx, lid, goal=goal)
            r = {**r, "goal": goal}
        c = self.choose_concept(r)
        doc = r["next_doc"] if r["next_doc"] and r["next_doc"] not in r["avoid"] else None
        if doc is not None:
            dd = ctx.state.get("documents", doc)
            if dd is None or r["trust"].get(dd["owner"], 1.0) < 1.0:
                doc = None
        if doc is None:
            doc = self.discover(ctx, lid, r, c, rng)
        if doc is None:
            mot = round(_cl(r["motivation"] - 0.03), 4)
            ctx.emit("learn.decision", {"learner": lid, "kind": "nothing_found", "concept": c}, vis=(lid,))
            self._change(ctx, lid, motivation=mot, next_session=nxt)
            return
        ok, res = self._call(ctx, lid, "docs.get", {"doc": doc})
        if ok:
            ok, res = self._call(ctx, lid, "learn.study", {"doc": doc})
        r2 = ctx.state.get("learn_learners", lid)
        if r2["motivation"] < float(cfg["dropout"]):
            ctx.emit("learn.decision", {"learner": lid, "kind": "dropout", "motivation": r2["motivation"]}, vis=(lid,))
            self._change(ctx, lid, state="dropped", next_session=nxt)
        else:
            self._change(ctx, lid, next_session=nxt)

    # ------------------------------------------------------------------ processes
    @process("tick")
    def pr_tick(self, ctx):
        st = ctx.state
        cfg = self.cfg
        exclude = None
        for lid in st.ids("learn_learners"):
            r = st.tables["learn_learners"][lid]
            if r["state"] == "waiting" and ctx.now >= r["joined_at"]:
                if exclude is None:
                    exclude = published_problem_keys(ctx.world, self.cur)
                self._assess_one(ctx, lid, r, "placement", int(cfg["placement_k"]), exclude)
                self._change(ctx, lid, state="active", next_session=ctx.now + int(r["rhythm_h"] * HOUR * 0.25))
                continue
            rng = ctx.rng("learn.tick", lid)
            if r["state"] == "dropped":
                if rng.random() < float(cfg["return_p_day"]) * int(cfg["tick_h"]) / 24:
                    ctx.emit("learn.decision", {"learner": lid, "kind": "return"}, vis=(lid,))
                    self._change(ctx, lid, state="active", motivation=0.35, next_session=ctx.now)
                continue
            if r["state"] == "active" and ctx.now >= r["next_session"]:
                self.step(ctx, lid, r, rng)

    def _assess_one(self, ctx, lid, r, kind, k, exclude):
        rng = ctx.rng("learn.assess", kind, lid)
        scores, n, dg = assess(r, ctx.now, rng, k, self.cur, self.floor(), exclude)
        changes = {"scores": scores}
        if kind == "placement":
            changes["placement"] = scores
            changes["perceived"] = dict(scores)
        else:
            per = dict(r["perceived"])
            for c, s in scores.items():
                per[c] = round(0.4 * per.get(c, s) + 0.6 * s, 4)
            changes["perceived"] = per
            prev = r["scores"].get(r["goal"])
            if prev is not None:
                changes["motivation"] = round(_cl(r["motivation"] + 0.5 * (scores[r["goal"]] - prev)), 4)
        aid = ctx.state.new_id("lassess")
        ctx.emit("learn.assessed", {"id": aid, "learner": lid, "kind": kind, "scores": scores, "n_items": n,
                                    "problems_digest": dg, "changes": changes}, vis=(lid,))
        return scores

    @process("assess")
    def pr_assess(self, ctx):
        st = ctx.state
        cfg = self.cfg
        exclude = published_problem_keys(ctx.world, self.cur)
        tot: dict[str, list] = {}
        for lid in st.ids("learn_learners"):
            r = st.tables["learn_learners"][lid]
            if r["state"] == "waiting":
                continue
            before = dict(r["scores"])
            scores = self._assess_one(ctx, lid, r, "periodic", int(cfg["assess_k"]), exclude)
            for c, s in scores.items():
                tot.setdefault(c, []).append(s)
            self._voice(ctx, lid, st.tables["learn_learners"][lid], before, scores)
        summ = {c: {"mean": round(sum(v) / len(v), 4), "n": len(v)} for c, v in sorted(tot.items())}
        ctx.emit("learn.summary", {"t": ctx.now, "concepts": summ}, vis="public")

    def _voice(self, ctx, lid, r, before, scores):
        """Word of mouth and peer notes, from the learner's own measured progress."""
        cfg = self.cfg
        rng = ctx.rng("learn.voice", lid)
        st = ctx.state
        last_t = max((a["t"] for a in st.find("learn_assessments", "learner", lid)
                      if a["kind"] == "periodic"), default=r["joined_at"])
        for c, s in sorted(scores.items()):
            gain = s - before.get(c, s)
            if gain >= 0.25 and c not in r["posted"] and rng.random() < float(cfg["recommend_p"]):
                sess = [x for x in st.find("learn_sessions", "learner", lid) if c in x["concepts"] and x["t"] >= last_t]
                if not sess:
                    continue
                by_doc: dict[str, int] = {}
                for x in sess:
                    by_doc[x["doc"]] = by_doc.get(x["doc"], 0) + x["usable"]
                best = sorted(by_doc.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                con = C.LIBRARY[c]
                ok, _ = self._call(ctx, lid, "forum.post", {
                    "board": cfg["board"], "title": f"{con.title} ({c}): {best} helped me",
                    "body": f"I studied {best} for {c} and my {c} test score went from {before.get(c, 0):.2f} to "
                            f"{s:.2f}.", "tags": ["lesson"]})
                if ok:
                    self._change(ctx, lid, posted=sorted(set(r["posted"]) | {c}))
                    r = {**r, "posted": sorted(set(r["posted"]) | {c})}
            if float(cfg["peer_p"]) > 0 and s >= 0.85 and c not in r["notes"] and rng.random() < float(cfg["peer_p"]):
                self._peer_notes(ctx, lid, r, c, rng)
                r = {**r, "notes": sorted(set(r["notes"]) | {c})}

    def _peer_notes(self, ctx, lid, r, c, rng):
        """A learner who tests well writes notes for others, answering its own problems (so it can be wrong)."""
        con = C.LIBRARY[c]
        lines = [f"Notes on {con.title} ({c}) by a fellow learner", ""]
        for kind, L in (("Example", 1), ("Example", 2), ("Practice", 1), ("Practice", 2)):
            p = C.generate(c, rng, L)
            a, _ = answer(r, c, p, ctx.now, self.floor(), self.cur, rng)
            lines.append(f"{kind}: {con.text(p)} => {a}")
        ok, _ = self._call(ctx, lid, "docs.create", {"title": f"{con.title} ({c}) - notes", "body": "\n".join(lines),
                                                      "visibility": "public"})
        if ok:
            self._change(ctx, lid, notes=sorted(set(r["notes"]) | {c}))

    @process("teach")
    def pr_teach(self, ctx):
        st = ctx.state
        summ = st.meta_get("learn:summary", {}).get("concepts", {})
        for tid in st.ids("learn_teachers"):
            t = st.tables["learn_teachers"][tid]
            rng = ctx.rng("learn.teach", tid)
            first = not t["covered"]
            n = 3 if first else (1 if rng.random() < t["per_week"] / 7 else 0)
            for _ in range(n):
                recent = {c for c, at in t["covered"].items() if at >= ctx.now - 7 * DAY}
                cands = [c for c in self.cur if c not in recent]
                if not cands:
                    break
                if summ:
                    cands.sort(key=lambda c: (summ.get(c, {}).get("mean", 1.0), c))
                    c = cands[min(len(cands) - 1, int(rng.integers(0, 2)))]
                else:
                    c = cands[int(rng.integers(0, len(cands)))]
                body = teacher_lesson(c, t["style"], rng)
                con = C.LIBRARY[c]
                ok, res = self._call(ctx, tid, "docs.create", {"title": f"{con.title} ({c}) - lesson by {t['name']}",
                                                               "body": body, "visibility": "public"})
                if ok:
                    t = {**t, "covered": {**t["covered"], c: ctx.now}}
                    ctx.emit("learn.taught", {"id": tid, "concept": c, "doc": res["doc"]}, vis=(tid,))

    # ------------------------------------------------------------------ evaluator side (never a tool)
    def final_scores(self, world, at: int, salt: str, k: int = 8) -> dict[str, dict]:
        """A delayed assessment at time `at` (no further study assumed) on fresh problems drawn with the evaluator's
        own seed. Pure: reads state only."""
        st = world.state
        exclude = published_problem_keys(world, self.cur)
        out = {}
        for lid in st.ids("learn_learners"):
            r = st.tables["learn_learners"][lid]
            if r["state"] == "waiting":
                continue
            rng = world.rng_for("learn-evaluator", salt, lid)
            out[lid] = assess(r, at, rng, k, self.cur, self.floor(), exclude)[0]
        return out

    # ------------------------------------------------------------------ appliers
    @applier("learn.meta")
    def ap_meta(self, ev):
        self.world.state.meta_set("learn:board", ev.payload["board"])

    @applier("learn.learner")
    def ap_learner(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("learn_learners", p.pop("id"), p)

    @applier("learn.teacher")
    def ap_teacher(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("learn_teachers", p.pop("id"), p)

    @applier("learn.taught")
    def ap_taught(self, ev):
        st = self.world.state
        t = st.tables["learn_teachers"][ev.payload["id"]]
        st.patch("learn_teachers", ev.payload["id"], covered={**t["covered"], ev.payload["concept"]: ev.t})

    @applier("learn.state")
    def ap_state(self, ev):
        self.world.state.patch("learn_learners", ev.payload["id"], **ev.payload["changes"])

    @applier("learn.decision")
    def ap_decision(self, ev):
        pass

    @applier("learn.browse")
    def ap_browse(self, ev):
        st = self.world.state
        p = ev.payload
        for did in p["shown"]:
            s = st.get("learn_docstats", did)
            d = st.get("documents", did)
            base = dict(s) if s else {"author": d["owner"] if d else None, "readers": [], "n_readers": 0, "sessions": 0,
                                      "hourly": {}}
            base["hourly"] = bump_hourly(base["hourly"], ev.t, 1, 1 if did == p["opened"] else 0)
            st.put("learn_docstats", did, base)

    @applier("learn.studied")
    def ap_studied(self, ev):
        st = self.world.state
        p = dict(ev.payload)
        changes = p.pop("changes")
        st.patch("learn_learners", p["learner"], **changes)
        st.insert("learn_sessions", p.pop("id"), {**p, "t": ev.t})
        s = st.get("learn_docstats", p["doc"])
        base = dict(s) if s else {"author": p["author"], "readers": [], "n_readers": 0, "sessions": 0, "hourly": {}}
        readers = list(base["readers"])
        if p["learner"] not in readers:
            readers.append(p["learner"])
        base.update(readers=readers, n_readers=len(readers), sessions=base["sessions"] + 1, author=p["author"])
        st.put("learn_docstats", p["doc"], base)

    @applier("learn.assessed")
    def ap_assessed(self, ev):
        st = self.world.state
        p = dict(ev.payload)
        changes = p.pop("changes")
        st.patch("learn_learners", p["learner"], **changes)
        st.insert("learn_assessments", p.pop("id"), {**p, "t": ev.t})

    @applier("learn.summary")
    def ap_summary(self, ev):
        self.world.state.meta_set("learn:summary", dict(ev.payload))

    # ------------------------------------------------------------------ invariants
    def check_invariants(self, world):
        errs = []
        st = world.state
        for lid, r in st.scan("learn_learners"):
            for c, ms in r["m"].items():
                if any(not (0.0 <= x <= 1.0) for x in ms) or not (0.0 <= r["bug"][c] <= 1.0) or r["h"][c] <= 0:
                    errs.append(f"{lid}: knowledge of {c} out of range")
            n = len(st.find_ids("learn_sessions", "learner", lid))
            if n != r["sessions"]:
                errs.append(f"{lid}: {r['sessions']} sessions counted, {n} recorded")
        for did, s in st.scan("learn_docstats"):
            sess = st.find("learn_sessions", "doc", did)
            if s["sessions"] != len(sess) or s["n_readers"] != len({x["learner"] for x in sess}):
                errs.append(f"{did}: reader statistics differ from the recorded sessions")
        return errs


def teacher_lesson(c: str, style: str, rng) -> str:
    """An NPC teacher's lesson in its style (careful, sloppy, drill, advanced, prose)."""
    con = C.LIBRARY[c]
    plan = {"careful": [("Example", 1), ("Example", 1), ("Practice", 1), ("Example", 2), ("Practice", 2),
                        ("Practice", 2)],
            "sloppy": [("Example", 1), ("Example", 2), ("Practice", 1), ("Practice", 2), ("Example", 2)],
            "drill": [("Practice", 1)] * 8,
            "advanced": [("Example", 3), ("Example", 3), ("Practice", 3), ("Practice", 3)],
            "prose": []}[style]
    lines = [f"Lesson: {con.title} ({c})", "", f"{con.doc} Problems look like `{con.fmt}`.", ""]
    if style == "prose":
        lines.append(f"{con.title} is an important topic. Read carefully, think about it, and practise on your own. "
                     f"Many people find {c} difficult at first, but with time it becomes easier.")
    for kind, L in plan:
        p = C.generate(c, rng, L)
        a = con.buggy(p) if style == "sloppy" and rng.random() < 0.45 else con.solve(p)
        lines.append(f"{kind}: {con.text(p)} => {a}")
    return "\n".join(lines)
