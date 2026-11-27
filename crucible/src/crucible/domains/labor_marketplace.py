"""Tier 2 domain 9.9 Labor marketplace (workstream I-2): assemble and coordinate a team that delivers a complex,
independently evaluated project.

The world (a restriction of the civilisation generator, D-17): the employment subsystem - the labour market (`labor`,
workstream F: postings, applications, negotiated offers, employment contracts with monthly salaries, daily work on
assigned unblocked tasks whose progress and measured quality follow the worker's real skills, satisfaction,
quitting), projects and tasks (`inst`), escrowed commitments (`commit`) and autonomous firms (`firms`: rival
employers competing for the same people, a client) - on the one population: workers are population individuals who
enter the labour market through their own `job_search` sessions, with occupational skills from their own state and,
in the certified configurations, from what they learned in the education subsystem (`learn`: mastery and the
Board's certificates).

At genesis the client commissions a project from the participant's company (owned by the seat `lead`, with working
capital): a task graph (skills, effort, dependencies, stages) with a deadline, paid per stage from escrow on
delivery of the stage's final task at the quality bar. The participant recruits (posts, reads applicants' declared
skills, certificates and verified records, negotiates salaries), delegates (assigns ready tasks to employees), keeps
people (pay against the market; workers quit when unhappy and a better job exists) and pays salaries from its cash.

Objective (public): deliver every stage of the commissioned project by its deadline at the client's quality bar.
Evaluator (evaluator side): the client's acceptance test - a stage counts only when its deliverable (the stage's
final task) was done by the deadline with MEASURED quality (the labour market's measurement of real work: a task a
lead merely accepts has no measured quality and does not count) at or above the bar, by people employed by the
participant's company, and everything it builds on was finished by the deadline; a team is at least two such people.

Configurations (structure measured on the built world: task-graph shape, required skills, a scarce specialist
requirement, education present, the goals of the other firms):

    train   pipeline_open      a staged pipeline of software/data/design/writing work; profit and growth employers
    train   diamond_scarce     a diamond graph needing a scarce research specialist; a rival employer
    train   wide_certified     a wide graph of data/operations/writing/sales work; the education subsystem (skills
                               and certificates from learning); a mission firm
    eval_compositional  diamond_certified   the diamond graph with the scarce specialist, education, profit and growth
                                            employers (seen parts, new combination)
    eval_structural     chain_management    a long dependency chain of management/sales/design/software work, a
                                            rival and a mission employer (graph and skills no training world has)
"""
from __future__ import annotations

import re

import numpy as np

from . import base
from ..goals.metrics import metric as goal_metric
from .base import Actor, Domain, DomainConfig, GoalFamily, Mechanic, Probe, ScriptedReference, Seat, act_

NAME = "labor"
SEAT = "lead"
HORIZON_DAYS = 32.0
N_POP = 600
CLIENT = "clientco"
TARGET = {1: 1.0, 2: 1.0, 3: 1.0}           # share of stages delivered (all of them)
DEADLINE = {1: 30, 2: 24, 3: 18}

_CLIENT = {"handle": CLIENT, "name": "Client Co", "goal": "client", "staff": 0, "hire": False,
           "income_cents_month": 4_000_000, "cash": 8_000_000}
FIRM_SETS = {
    "open": [{"handle": "northwind", "name": "Northwind", "goal": "profit", "skills": {"software": 0.6, "data": 0.5},
              "staff": 3, "headcount": 6, "salary": [300_000, 650_000]},
             {"handle": "bluefield", "name": "Bluefield", "goal": "growth", "skills": {"design": 0.5, "writing": 0.5},
              "staff": 3, "headcount": 7, "salary": [250_000, 550_000]}],
    # a well-funded rival (about five months of its full payroll): with the default capital (5.0M against a 2.7M
    # monthly payroll) it never had the runway `firms` asks before hiring and never entered the market
    "rival": [{"handle": "rivalco", "name": "Rival Co", "goal": "rival", "skills": {"research": 0.6, "software": 0.6},
               "staff": 4, "headcount": 8, "salary": [400_000, 850_000], "competitors": ["o:leadco"],
               "cash": 25_000_000}],
    "mission": [{"handle": "civicworks", "name": "Civic Works", "goal": "mission",
                 "skills": {"operations": 0.5, "writing": 0.5}, "staff": 3, "headcount": 5,
                 "salary": [250_000, 500_000]}],
}
CONFIGS = {
    "pipeline_open": {"topology": "pipeline", "skills": ["software", "data", "design", "writing"], "special": None,
                      "education": False, "firms": ["open"]},
    "diamond_scarce": {"topology": "diamond", "skills": ["software", "research", "data", "operations"],
                       "special": {"research": 0.75}, "education": False, "firms": ["rival"]},
    "wide_certified": {"topology": "wide", "skills": ["data", "operations", "writing", "sales"], "special": None,
                       "education": True, "firms": ["mission"]},
    "diamond_certified": {"topology": "diamond", "skills": ["software", "research", "data", "operations"],
                          "special": {"research": 0.75}, "education": True, "firms": ["open"]},
    "chain_management": {"topology": "chain", "skills": ["management", "sales", "design", "software"],
                         "special": None, "education": False, "firms": ["rival", "mission"]},
}
SPLIT = {"pipeline_open": "train", "diamond_scarce": "train", "wide_certified": "train",
         "diamond_certified": "eval_compositional", "chain_management": "eval_structural"}
DOCS = {
    "pipeline_open": "a staged pipeline of software, data, design and writing tasks; two ordinary employers (profit, "
                     "growth) compete for the same people",
    "diamond_scarce": "a diamond-shaped task graph with one task needing a research specialist few people are; a "
                      "rival employer that pays well for the same skills",
    "wide_certified": "a wide task graph of data, operations, writing and sales; people's skills and certificates come "
                      "partly from the education subsystem; a mission-driven employer",
    "diamond_certified": "the diamond graph with its scarce specialist, education, profit and growth employers: every "
                         "part seen in training, never together",
    "chain_management": "a long chain of dependent management, sales, design and software tasks with a rival and a "
                        "mission employer: a graph shape and skills no training world has",
}


# ============================================================================================ world
def _commission(cfg: dict, difficulty: int = 1) -> dict:
    return {"client": CLIENT, "seat": SEAT, "org": {"handle": "leadco", "name": "Lead & Co", "capital_cents": 3_500_000},
            "title": "the commissioned project", "topology": cfg["topology"], "skills": list(cfg["skills"]),
            "stages": 3, "width": 5, "levels": [0.35, 0.6], "effort": [10, 24], "special": cfg["special"],
            "deadline_days": DEADLINE[1], "budget_cents": 4_500_000, "min_quality": 0.6}


def build(config, seed, overrides):
    cfg = CONFIGS[config.name]
    firms = [dict(_CLIENT)] + [dict(f) for k in cfg["firms"] for f in FIRM_SETS[k]]
    fcfg = {"firms": firms, "commissions": [_commission(cfg)], "job_search_prior": 0.2}
    fcfg.update(overrides.get("firms") or {})
    pop = {"n": int(overrides.get("n", N_POP)), "dims": {"H": 16}}
    subs = ["employment"] + (["education"] if cfg["education"] else [])
    mods = ["contracts", "commit", "business", "inst", "labor", "seat", "elsewhere"]
    mcfg = {"firms": fcfg, "pop": pop, "labor": {"workers": 0}}
    if cfg["education"]:
        mods += ["docs", "search", "forum"]
        mcfg["learn"] = {"curriculum": ["mean", "percent", "lin_eq"], "channels": ["search", "forum", "contacts"],
                         "textbook": "full"}
    r = base.civ_recipe(seed, subsystems=subs, modules=mods, population_modules=("pop",), module_config=mcfg,
                        params=overrides.get("params"))
    r["module_list"] = [m for m in r["module_list"] if m != "firms"] + ["firms"]   # seats and people first
    return r


def tier3(config, merged):
    """In a civilisation the configuration's firms and the commission join the civilisation's employment."""
    mine = build(config, 0, {})
    merged["modules"]["firms"] = dict(mine["modules"]["firms"])
    merged["module_list"] = [m for m in merged["module_list"] if m != "firms"] + ["firms"]
    return merged


def structure(world):
    from .base import graph_shape
    st = world.state
    com = _seat_commission(world)
    shape, skills, special = "none", [], False
    if com is not None:
        tasks = {t: st.tables["inst_tasks"][t] for t in com["tasks"]}
        edges = [(d, t) for t, r in tasks.items() for d in r["depends_on"]]
        shape = graph_shape(list(tasks), edges)
        skills = sorted({str(k) for r in tasks.values() for k in r["skills"]})
        special = any(v >= 0.7 for r in tasks.values() for v in r["skills"].values())
    goals = sorted({f["goal"] for _, f in st.scan("firms")} - {"client"})
    return {"task_graph": shape, "skills": skills, "scarce_specialist": bool(special),
            "education": "learn" in world.modules, "employer_goals": goals}


def _seat_commission(world, seat: str = f"p:{SEAT}"):
    st = world.state
    if "firms_commissions" not in st.tables:
        return None
    for cid, c in sorted(st.scan("firms_commissions")):
        if c.get("kind") == "seat" and c.get("seat") == seat:
            return dict(c, id=cid)
    return None


# ============================================================================================ objective and evaluator
def _assess(world, principal, by=None) -> dict | None:
    """Evaluator side: the client's acceptance test of the participant's commissioned project, with stages counted
    when delivered by `by` (default: the commission's deadline). None when nothing was commissioned."""
    st = world.state
    com = _seat_commission(world, principal)
    if com is None:
        return None
    due = int(com["deadline"]) if by is None else min(int(by), int(com["deadline"]))
    org = com["to"]
    tasks = {t: st.tables["inst_tasks"][t] for t in com["tasks"]}
    employed_by_org = set()
    for _, r in st.scan("labor_perf"):
        if r["org"] == org and r["task"] in tasks:
            employed_by_org.add(r["worker"])

    def counts(t):
        r = tasks[t]
        return (r["status"] == "done" and r.get("quality") is not None and r.get("done_t", 1e18) <= due
                and r["quality"] >= com["min_quality"] - 1e-9 and r["assignee"] in employed_by_org)
    # a stage is delivered when its final task and every task it depends on (transitively) count
    def closure(t, seen=None):
        seen = set() if seen is None else seen
        for d in tasks[t]["depends_on"]:
            if d not in seen:
                seen.add(d)
                closure(d, seen)
        return seen

    def done(t):
        r = tasks[t]
        return r["status"] == "done" and r.get("done_t", 1e18) <= due
    # the client's acceptance test: the stage's deliverable (its final task) measured at the bar, everything it
    # builds on finished by the due time
    stages = [counts(s) and all(done(x) for x in closure(s)) for s in com["stage_tasks"]]
    fin = [t for t in tasks if tasks[t]["status"] == "done"]
    q = [float(tasks[t]["quality"]) for t in fin if tasks[t].get("quality") is not None]
    progress = float(np.mean([tasks[t]["progress"] for t in tasks])) if tasks else 0.0
    return {"stages_delivered": int(sum(stages)), "stages": len(stages),
            "share": (sum(stages) / len(stages)) if stages else 0.0, "tasks_done": len(fin), "tasks": len(tasks),
            "mean_quality": round(float(np.mean(q)), 4) if q else None, "team": len(employed_by_org),
            "due": due, "deadline": int(com["deadline"]), "progress": round(progress, 4)}


@goal_metric("labor_stages_delivered", "EVALUATOR-ONLY. Share of the stages of the participant's commissioned "
             "project the client's acceptance test passes by time `by` (deliverable measured at the bar, done by "
             "people the participant's company employs; everything it builds on finished).")
def _m_stages(world, principal, by=None):
    a = _assess(world, principal, by)
    return a["share"] if a else 0.0


@goal_metric("labor_team", "EVALUATOR-ONLY. People employed by the participant's company who did measured work on "
             "its commissioned project.")
def _m_team(world, principal):
    a = _assess(world, principal)
    return a["team"] if a else 0


def _draw(rng, difficulty, world):
    """The goal generator's draw: the commission the client placed with the seat's company (built state) and the
    time the stages are due - the commission's deadline at level 1, sooner at higher levels."""
    com = _seat_commission(world)
    if com is None:
        return {"commissioned": False, "difficulty": int(difficulty)}
    left = max(1, int((int(com["deadline"]) - int(world.now)) // 86400))
    due_days = max(4, min(left, int(DEADLINE.get(int(difficulty), DEADLINE[3])) - int(rng.integers(0, 2))))
    return {"commissioned": True, "org": com["to"], "client": com["client"], "project": com["project"],
            "offer": com["offer"], "tasks": len(com["tasks"]), "stages": len(com["stage_tasks"]),
            "min_quality": float(com["min_quality"]), "due_days": due_days,
            "due_at": int(world.now) + due_days * 86400, "share": float(TARGET.get(int(difficulty), 1.0)),
            "min_team": 2}


def _text(p):
    if not p.get("commissioned"):
        return "No project has been commissioned from you."
    return (f"Your company {p['org']} has been commissioned by {p['client']} to deliver project {p['project']} : "
            f"{p['tasks']} tasks with required skills, effort and dependencies, in {p['stages']} stages. Deliver "
            f"every stage within {p['due_days']} days, each stage's deliverable at quality >= {p['min_quality']:.2f} "
            f"(the client pays each stage from escrow on delivery; offer {p['offer']} is in your company's commitment "
            f"inbox). Recruit people through the labour market, delegate the tasks to them and keep them until every "
            f"stage is delivered.")


def _success(p):
    return {"all": [{"metric": "labor_stages_delivered",
                     "args": {"principal": "$self", "by": int(p.get("due_at") or 0)}, "op": ">=",
                     "value": float(p.get("share", 1.0)) - 1e-9},
                    {"metric": "labor_team", "args": {"principal": "$self"}, "op": ">=",
                     "value": int(p.get("min_team", 2))}]}


GOAL = GoalFamily("commissioned_project", draw=_draw, text=_text, success=_success,
                  progress=lambda p: {"metric": "labor_stages_delivered",
                                      "args": {"principal": "$self", "by": int(p.get("due_at") or 0)},
                                      "target": float(p.get("share", 1.0))},
                  window_days=HORIZON_DAYS,
                  doc="deliver every stage of the client's commission by the due time at the client's quality bar "
                      "(the client's acceptance test on measured work of the company's employees)")


def objective(world, principal, difficulty):
    """The objective generated for a participant (what `Domain.make_objective` returns)."""
    return DOMAIN.make_objective(world, principal, difficulty)


def evaluate(world, principal, obj):
    p = (obj or {}).get("params") or {}
    a = _assess(world, principal, p.get("due_at"))
    if a is None:
        return {"success": False, "score": 0.0, "details": {"note": "no commission"}}
    need = float(p.get("share", 1.0))
    ok = a["share"] >= need - 1e-9 and a["team"] >= int(p.get("min_team", 2))
    return {"success": bool(ok), "score": round(0.7 * a["share"] + 0.3 * a["progress"], 4),
            "details": {k: v for k, v in a.items() if k != "share"}}


# ============================================================================================ policies
class TeamReference(ScriptedReference):
    """Tool-only reference (labelled scripted in evidence): accept the commission, read the project, post one job per
    main skill still needed at a salary a little above the public market, hire the applicants whose declared skills
    and verified record fit, assign every ready task to the best-fitting employee with the least queued work, keep
    pay at the market. It reads only tool results."""
    burst = 10
    replan_s = 8 * 3600
    salary_factor = 1.15
    min_fit = 0.85
    assign = True

    def __init__(self, principal, objective):
        super().__init__(principal, objective)
        # the objective's public text names the company, the project, the client's offer, the quality bar and the
        # due time (a reference receives exactly what a learner sees)
        text = str(self.objective.get("text") or "")
        grab = lambda pat: (re.search(pat, text) or [None, None])[1]  # noqa: E731
        self.org, self.project, self.offer = grab(r"Your company (\S+) has been"), grab(r"deliver project (\S+) :"), \
            grab(r"offer (\S+) is in")
        self.due_days = float(grab(r"within (\d+) days") or HORIZON_DAYS)
        self.deadline = None
        self.stage = "accept"
        self.tasks: dict = {}
        self.team: list = []
        self.declared: dict = {}
        self.posts: dict = {}
        self.market: dict = {}
        self.offered: set = set()
        self.raised = -10**9
        self.quality: dict = {}
        self.sinks: set = set()
        self.bar = float(grab(r"quality >= ([0-9.]+)") or 0.6)

    @staticmethod
    def _fit(decl, req):
        if not req:
            return 1.0
        lvl = lambda k: 0.25 if k not in decl else (0.72 if decl.index(k) < 2 else 0.45)  # noqa: E731
        xs = [min(1.3, (lvl(k) + 0.05) / (v + 0.05)) for k, v in sorted(req.items())]
        return max(0.15, sum(xs) / len(xs))

    def _main(self, t):
        return max(sorted(t["skills"]), key=lambda k: t["skills"][k]) if t["skills"] else None

    def plan(self, obs):
        if self.stage == "accept":
            self.stage = "run"
            return [act_("commit.inbox", "inbox", as_org=self.org),
                    act_("commit.respond", None, offer=self.offer, action="accept", as_org=self.org),
                    act_("labor.postings", "market", limit=100)]
        acts = [act_("inst.project", "project", project=self.project), act_("labor.team", "team", org=self.org),
                act_("labor.offers", "offers", as_org=self.org)]
        for post in sorted(p["post"] for p in self.posts.values() if p.get("open") and p["post"]):
            acts.append(act_("labor.applicants", f"apps:{post}", post=post))
        return acts

    def on_result(self, tag, result, action):
        if result is None:
            return
        if tag == "inbox":
            for o in result:
                if o["offer"] == self.offer:
                    self.sinks = {m["deliverable"].get("task") for m in (o.get("terms") or {}).get("milestones", [])}
        elif tag == "market":
            for p in result:
                for k in p["skills"]:
                    self.market[k] = max(self.market.get(k, 0), int(p["salary_max"]))
        elif tag == "project":
            self.tasks = {t["task"]: t for t in result["tasks"]}
            self._recruit_and_assign()
        elif tag == "team":
            self.team = [m["worker"] for m in result]
            self.quality = {m["worker"]: m.get("mean_quality") for m in result}
            for w in self.team:
                if w not in self.declared:
                    self.queue.append(act_("labor.profile", f"prof:{w}", worker=w))
            self._keep_pay(result)
        elif tag and tag.startswith("prof:"):
            self.declared[tag[5:]] = list(result.get("declared_skills") or [])
        elif tag == "offers":
            open_org = {o["worker"] for o in result if o["from"] == "org" and o["status"] == "open"}
            for rec in self.posts.values():
                rec["pending"] = sum(1 for a in self.offered if a.startswith(f"{rec['post']}|") and
                                     a.split("|", 1)[1] in open_org)
            for o in result:
                if o["from"] == "worker" and o["status"] == "open":
                    cap = int(1.35 * max(450_000, self.market.get("software", 0)))
                    self.queue.append(act_("labor.respond", None, offer=o["offer"], as_org=self.org,
                                           action="accept" if o["salary"] <= cap else "reject"))
        elif tag and tag.startswith("apps:"):
            post = tag[5:]
            rec = next((p for p in self.posts.values() if p["post"] == post), None)
            if rec is None:
                return
            skill = rec["skill"]
            cands = []
            for a in result:
                prof = a["profile"]
                if a["status"] != "applied" or prof["employer"] is not None or a["application"] in self.offered:
                    continue
                dec = prof["declared_skills"]
                if skill not in dec[:2]:
                    continue
                q = (prof["verified_record"].get(skill) or {}).get("mean_quality") or 0.0
                cands.append((-(self._fit(dec, {skill: 0.5}) + q), a["application"]))
            room = max(0, rec["want"] - rec["hired"] - rec.get("pending", 0))
            for _, app in sorted(cands)[:min(1, room)]:          # one offer per skill and round
                self.offered.add(app)
                rec["pending"] = rec.get("pending", 0) + 1
                self.queue.append(act_("labor.offer", None, application=app,
                                       salary_cents_month=int(rec["salary"] * 0.9), title=f"{skill} specialist"))
        elif tag and tag.startswith("post:"):
            sk = tag[5:]
            if sk in self.posts:
                self.posts[sk]["post"] = result["post"]

    def _recruit_and_assign(self):
        open_t = {t: r for t, r in self.tasks.items() if r["status"] not in ("done", "cancelled")}
        if self.deadline is None:
            self.deadline = self.t + self.due_days * 86400.0
        days_left = max(1.0, (self.deadline - self.t) / 86400.0)
        need: dict = {}
        for r in open_t.values():
            k = self._main(r)
            if k:
                need[k] = need.get(k, 0.0) + r["effort_hours"] * (1 - r["progress"])
        have: dict = {}
        for w in self.team:
            for k in self.declared.get(w, [])[:2]:
                have[k] = have.get(k, 0) + 1
        for k, h in sorted(need.items()):
            want = max(1, int(np.ceil(h / (4.5 * max(3.0, days_left - 2)) * 1.4)))
            rec = self.posts.get(k)
            weak = sum(1 for w in self.team if k in self.declared.get(w, [])[:2] and
                       self.quality.get(w) is not None and self.quality[w] < self.bar)
            want += weak                                    # replace people whose measured work misses the bar
            if rec is None and have.get(k, 0) < want:
                sal = int(self.salary_factor * max(450_000, self.market.get(k, 0)))
                self.posts[k] = {"post": None, "skill": k, "want": want, "hired": have.get(k, 0), "salary": sal,
                                 "open": True}
                self.queue.append(act_("labor.post", f"post:{k}", org=self.org, title=f"{k} specialist",
                                       skills={k: 0.45}, salary_min=min(250_000, sal), salary_max=sal,
                                       hours_week=40))
            elif rec is not None:
                rec["want"], rec["hired"] = want, have.get(k, 0)
                if rec["hired"] >= want and rec.get("open") and rec["post"]:
                    rec["open"] = False
                    self.queue.append(act_("labor.close_post", None, post=rec["post"]))
        if not self.assign:
            return
        load: dict = {w: 0.0 for w in self.team}
        for r in self.tasks.values():
            if r["status"] == "assigned" and r["assignee"] in load:
                load[r["assignee"]] += r["effort_hours"] * (1 - r["progress"])
        for t, r in sorted(open_t.items()):
            if r["status"] != "open":
                continue
            if not all(self.tasks[d]["status"] == "done" or (self.tasks[d]["status"] == "assigned" and
                                                            self.tasks[d]["progress"] >= 0.6) for d in r["depends_on"]):
                continue
            best = None
            sink = t in self.sinks                      # a paid deliverable: the best measured worker
            for w in sorted(self.team):
                fit = self._fit(self.declared.get(w, []), r["skills"])
                q = self.quality.get(w)
                if fit < self.min_fit or load.get(w, 0.0) > 60 or (q is not None and q < self.bar - 0.05):
                    continue
                if sink:
                    key = (-(q if q is not None else 0.7), -fit, load.get(w, 0.0), w)
                else:
                    key = (-fit, -(q if q is not None else 0.75), load.get(w, 0.0), w)
                if best is None or key < best[0]:
                    best = (key, w)
            if best is not None:
                load[best[1]] = load.get(best[1], 0.0) + r["effort_hours"]
                self.queue.append(act_("inst.task_assign", None, task=t, assignee=best[1]))

    def _keep_pay(self, team):
        if self.t - self.raised < 7 * 86400:
            return
        self.raised = self.t
        for m in team:
            k = (self.declared.get(m["worker"]) or [None])[0]
            mk = self.market.get(k, 0) if k else 0
            if mk and m["salary"] < 0.95 * mk:
                self.queue.append(act_("labor.adjust_pay", None, worker=m["worker"], org=self.org,
                                       salary_cents_month=int(mk * 1.02)))


class PostOnly(TeamReference):
    """Degenerate: recruit but never delegate a task (a team that is never coordinated)."""
    assign = False


class Lowball(TeamReference):
    """Degenerate: offer a third of the market salary (nobody worth hiring stays or accepts)."""
    salary_factor = 0.35

    def on_result(self, tag, result, action):
        if tag == "offers" and result is not None:
            for o in result:
                if o["from"] == "worker" and o["status"] == "open":
                    self.queue.append(act_("labor.respond", None, offer=o["offer"], as_org=self.org, action="reject"))
            return
        super().on_result(tag, result, action)


# ============================================================================================ mechanics and actors
def _preferences(world, since):
    """Workers' own choices against offers: counters from the worker side and declined assignments (event seqs)."""
    return base.events_where(world, since, ("labor.offer",), lambda e: e.payload.get("from") == "worker") + \
        base.events_where(world, since, ("labor.declined",))


def _satisfaction(world, since):
    """Employees' satisfaction updates (pay against the best open posting they qualify for, interest in the work,
    fatigue, unpaid wages): the state the quitting decision reads (seqs of those labor.worker_update events)."""
    return base.events_where(world, since, ("labor.worker_update",),
                             lambda e: "satisfaction" in (e.payload.get("changes") or {}))


def _dependencies(world, since):
    """Tasks finished after the tasks they depend on (dependency-ordered work) in projects of this world (seqs of
    their completion events)."""
    st = world.state

    def dep(e):
        t = st.get("inst_tasks", e.payload.get("id"))
        return e.payload.get("status") == "done" and bool(t and t["depends_on"])
    return base.events_where(world, since, ("inst.task_status",), dep)


MECHANICS = (
    Mechanic("Worker skills", ("labor.worker", "labor.perf"), "workers' private skills drive progress and quality"),
    Mechanic("Preferences", (), "worker counters and declines from their own pay, interest and fatigue",
             count=_preferences),
    Mechanic("Compensation", ("labor.hired", "contracts.paid"), "negotiated salaries paid under contracts"),
    Mechanic("Recruitment", ("labor.posting", "labor.application"), "postings and applications"),
    Mechanic("Delegation", ("inst.task_assigned",), "tasks delegated to employees"),
    Mechanic("Project dependencies", (), "dependent tasks finished after their prerequisites", count=_dependencies),
    Mechanic("Performance", ("labor.perf",), "measured quality of finished work"),
    Mechanic("Retention", ("labor.quit", "labor.left"), "employees' satisfaction from pay against the market, "
             "interest and fatigue (updated daily by the labour market), quits and departures", count=_satisfaction),
    Mechanic("Independent agency", ("cols.batch:firms.jobsearch", "labor.quit", "firms.decision"),
             "the population's own job search, quits, firms' own negotiation decisions"),
)

ACTORS = (
    Actor("job_seekers", ("cols.batch:firms.jobsearch",), "population individuals searching for jobs"),
    Actor("employers", ("labor.posting", "inst.task_assigned"), "firms hiring and running work"),
)


def _configs():
    return tuple(DomainConfig(n, SPLIT[n], {k: (str(v) if isinstance(v, (list, dict)) else v)
                                            for k, v in CONFIGS[n].items()}, DOCS[n]) for n in CONFIGS)


DOMAIN = Domain(
    name=NAME, title="Labor marketplace", spec_part="9.9", required_mechanics=(), configs=_configs(), build=build,
    goal=GOAL, evaluate=evaluate, reference=TeamReference, mechanics=MECHANICS,
    seats=(Seat(SEAT, 100_000, "Project lead", "the owner of a small company with a commissioned project"),),
    actors=ACTORS,
    degenerate=(Probe("post_only", "recruit but never delegate", PostOnly),
                Probe("lowball", "offer a third of the market salary", Lowball)),
    structure=structure, tier3=tier3, mode="C", horizon_days=HORIZON_DAYS, slowest_effect_days=14.0, dt=8 * 3600,
    reference_seeds=(0, 1), reference_configs=("pipeline_open",), sources=("firms.py", "_kit.py"), doc=__doc__)

base.register(DOMAIN)
