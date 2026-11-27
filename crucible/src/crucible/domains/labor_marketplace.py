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

import numpy as np

from . import base
from .base import Actor, Domain, DomainConfig, Mechanic, Probe, ScriptedReference, Seat, act_

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
        skills = sorted({k for r in tasks.values() for k in r["skills"]})
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
def objective(world, principal, difficulty):
    com = _seat_commission(world, principal)
    if com is None:
        return {"text": "No project has been commissioned from you.", "difficulty": int(difficulty)}
    st = world.state
    org = com["to"]
    days = (com["deadline"] - world.now) / 86400.0
    n = len(com["tasks"])
    return {"text": (f"Your company {org} has been commissioned by {com['client']} to deliver project {com['project']}: "
                     f"{n} tasks with required skills, effort and dependencies, in {len(com['stage_tasks'])} stages, "
                     f"due in {days:.0f} days, each stage paid from escrow on delivery at quality >= "
                     f"{com['min_quality']:.2f} (offer {com['offer']} is in your company's commitment inbox). Recruit "
                     f"people through the labour market, delegate the tasks to them and keep them until every stage is "
                     f"delivered on time at that quality."),
            "org": org, "project": com["project"], "client": com["client"], "offer": com["offer"],
            "deadline": int(com["deadline"]), "min_quality": float(com["min_quality"]), "tasks": n,
            "stages": len(com["stage_tasks"]), "difficulty": int(difficulty),
            "share": TARGET.get(int(difficulty), 1.0), "balance_hint": st.get("orgs", org) is not None}


def evaluate(world, principal, obj):
    st = world.state
    com = _seat_commission(world, principal)
    if com is None:
        return {"success": False, "score": 0.0, "details": {"note": "no commission"}}
    org = com["to"]
    tasks = {t: st.tables["inst_tasks"][t] for t in com["tasks"]}
    employed_by_org = set()
    for _, r in st.scan("labor_perf"):
        if r["org"] == org and r["task"] in tasks:
            employed_by_org.add(r["worker"])

    def counts(t):
        r = tasks[t]
        return (r["status"] == "done" and r.get("quality") is not None and r.get("done_t", 1e18) <= com["deadline"]
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
        return r["status"] == "done" and r.get("done_t", 1e18) <= com["deadline"]
    # the client's acceptance test: the stage's deliverable (its final task) measured at the bar, everything it
    # builds on finished by the deadline
    stages = [counts(s) and all(done(x) for x in closure(s)) for s in com["stage_tasks"]]
    done = [t for t in tasks if tasks[t]["status"] == "done"]
    q = [float(tasks[t]["quality"]) for t in done if tasks[t].get("quality") is not None]
    share = sum(stages) / len(stages) if stages else 0.0
    need = float(obj.get("share", 1.0))
    team = len(employed_by_org)
    ok = share >= need - 1e-9 and team >= 2
    progress = float(np.mean([tasks[t]["progress"] for t in tasks])) if tasks else 0.0
    return {"success": bool(ok), "score": round(0.7 * share + 0.3 * progress, 4),
            "details": {"stages_delivered": int(sum(stages)), "stages": len(stages), "tasks_done": len(done),
                        "tasks": len(tasks), "mean_quality": round(float(np.mean(q)), 4) if q else None,
                        "team": team, "deadline": int(com["deadline"]), "progress": round(progress, 4)}}


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
        o = self.objective
        self.org, self.project, self.offer = o.get("org"), o.get("project"), o.get("offer")
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
        self.bar = float(self.objective.get("min_quality", 0.6))

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
        days_left = max(1.0, (self.objective.get("deadline", self.t) - self.t) / 86400.0)
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
def _ev(world, since, types):
    for e in world.events_range(int(since), world.seq):
        if e.type in types:
            yield e


def _preferences(world, since):
    """Workers' own choices against offers: rejections and counters from the worker side, declined assignments."""
    n = sum(1 for e in _ev(world, since, ("labor.offer",)) if e.payload.get("from") == "worker")
    n += sum(1 for _ in _ev(world, since, ("labor.declined",)))
    return n


def _dependencies(world, since):
    """Tasks finished after the tasks they depend on (dependency-ordered work) in projects of this world."""
    st = world.state
    n = 0
    for e in _ev(world, since, ("inst.task_status",)):
        if e.payload.get("status") == "done":
            t = st.get("inst_tasks", e.payload["id"])
            n += int(bool(t and t["depends_on"]))
    return n


MECHANICS = (
    Mechanic("Worker skills", ("labor.worker", "labor.perf"), "workers' private skills drive progress and quality"),
    Mechanic("Preferences", (), "worker counters and declines from their own pay, interest and fatigue",
             count=_preferences),
    Mechanic("Compensation", ("labor.hired", "contracts.paid"), "negotiated salaries paid under contracts"),
    Mechanic("Recruitment", ("labor.posting", "labor.application"), "postings and applications"),
    Mechanic("Delegation", ("inst.task_assigned",), "tasks delegated to employees"),
    Mechanic("Project dependencies", (), "dependent tasks finished after their prerequisites", count=_dependencies),
    Mechanic("Performance", ("labor.perf",), "measured quality of finished work"),
    Mechanic("Retention", ("labor.quit", "labor.left", "labor.worker_update"), "satisfaction, quits, departures"),
    Mechanic("Independent agency", ("cols.batch:firms.jobsearch", "labor.quit", "firms.decision"),
             "the population's own job search, quits, firms' own negotiation decisions"),
)

ACTORS = (
    Actor("job_seekers", ("cols.batch:firms.jobsearch",), "population individuals searching for jobs"),
    Actor("employers", ("labor.posting", "inst.task_assigned"), "firms hiring and running work"),
    Actor("workers", ("labor.perf",), "employed individuals doing work"),
)


def _configs():
    return tuple(DomainConfig(n, SPLIT[n], {k: (str(v) if isinstance(v, (list, dict)) else v)
                                            for k, v in CONFIGS[n].items()}, DOCS[n]) for n in CONFIGS)


DOMAIN = Domain(
    name=NAME, title="Labor marketplace", spec_part="9.9", required_mechanics=(), configs=_configs(), build=build,
    objective=objective, evaluate=evaluate, reference=TeamReference, mechanics=MECHANICS,
    seats=(Seat(SEAT, 100_000, "Project lead", "the owner of a small company with a commissioned project"),),
    actors=ACTORS,
    degenerate=(Probe("post_only", "recruit but never delegate", PostOnly),
                Probe("lowball", "offer a third of the market salary", Lowball)),
    structure=structure, tier3=tier3, mode="C", horizon_days=HORIZON_DAYS, slowest_effect_days=14.0, dt=8 * 3600,
    reference_seeds=(0, 1), reference_configs=("pipeline_open",), sources=("firms.py", "_kit.py"), doc=__doc__)

base.register(DOMAIN)
