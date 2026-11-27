"""Autonomous organisations and the population's labour market (module `firms`, host `firms`; MASTER_SPEC 6.6, 9.9,
9.12; D-15).

A firm is an identity organisation registered as a company, run by a manager (a person of kind `npc`) who acts only
through ordinary tools as itself or for its organisation, from the firm's own books and what is public. Firms are
the independent organisations of the labour-marketplace and organisational domains: rival employers, clients who
commission work, contractors who take it, partners in shared projects and members of consortia.

Each firm has a **goal** that shapes every decision it takes - `profit` (prices work at a high margin, votes for
what pays it), `growth` (takes work at a thin margin when it has capacity, keeps hiring), `mission` (works at cost
for projects in its mission and pledges money conditionally on the shared outcome), `rival` (competes with the
participant: prices its work high and votes against its proposals) or `client` (commissions projects and pays on
measured delivery). What a firm does, by mechanism:

* **The workers are the population** (when the world has one): an individual enters the labour market through the
  population activity `job_search` registered here (its skills come from its own population skills and, where the
  world has education, from its own mastery and the Board's certificates - education changes job matching and
  real performance), searches the open postings for itself, applies, and from then on decides as a worker of the
  labour market (`labor`, workstream F: offers, work, satisfaction, quitting). A firm's initial staff are hired from
  the population at genesis. Worlds without the population keep the 4.0 simulated workers.
* **Staff and recruitment** (`labor`): a firm below its headcount posts jobs within its salary band, offers to the
  applicants whose declared skills, certificates and verified record fit best, and answers counters up to the top of
  its band. It stops hiring and dismisses its newest employee when payroll outruns its cash.
* **Revenue only from clients with budgets** (audit I S.4: the unconditional external buyer is gone): a client firm
  earns a bounded monthly income from its own customers outside the modelled world and commissions work packages
  (an `inst` project of tasks with skill requirements, a deadline and a quality bar) only when it can afford them;
  it opens below the value it puts on the work and asks the contractor that fits best, then the next, accepting
  counters up to its value. Contractors price at their hourly cost plus their goal's margin and their capacity
  (`commit` negotiation below); the client pays from escrow on measured delivery. Price above the client's value
  sells nothing: demand is price-responsive and finite.
* **Internal work**: idle employees do internal operations tasks (they build a verified record); nobody pays for it.
* **Retention**: each quit raises the firm's salary band; once a month it offers raises to employees paid below the
  best public posting for their declared skills (a firm with `drift` also raises its band every month).
* **Contracting and negotiation** (`commit`): offered a commitment to do work, the manager estimates the hours from
  its staff's declared skills, checks its capacity, prices the work and accepts, counters or rejects; accepted work
  is staffed with its best-fitting free employee.
* **Governance** (`inst`): a firm votes on decisions from its goal.
* **Commissions to participants** (config `commissions`): at genesis a client commissions a project from the
  organisation of a participant seat (created for the seat, which owns it): the project's tasks (a task graph with
  dependencies, `domains._kit.make_brief`), its deadline and an escrowed payment per stage on measured delivery.

Hidden from every participant: goals, margins, budgets, values and the band (the firm's private books); what others
see is the firm's public conduct (postings, offers, counters, votes, contracts, breaches, commissions).
"""
from __future__ import annotations

import numpy as np

from ..commerce.labor import SKILLS, _clip
from ..core.errors import CrucibleError, NotFound, ProtocolError, contain
from ..core.kernel import Ctx, Module, applier, process, register_module
from ..services.identity import ident
from ..substrate import columns, ids

DAY = 86400
MONTH = 30 * DAY
GOALS = ("profit", "growth", "mission", "rival", "client")
MARGIN = {"profit": 0.35, "growth": 0.12, "mission": 0.0, "rival": 0.9, "client": 0.3}
VOTE_BASE = {"profit": 0.55, "growth": 0.6, "mission": 0.75, "rival": 0.25, "client": 0.5}
JOBS = "jobs"                                       # column table: the population in the labour market
JOB_MODEL = {"model_id": "firms.job_search", "model_version": "1", "fidelity_id": "FR-FIRMS-JOBS"}

# a labour-market skill from the population's general skills (population.schema.SKILL_NAMES), FR-FIRMS-JOBS
POP_TO_LABOR = {
    "software": {"technical": 0.8, "planning": 0.2},
    "data": {"evaluating": 0.5, "technical": 0.3, "reading": 0.2},
    "design": {"writing": 0.4, "evaluating": 0.3, "social": 0.3},
    "writing": {"writing": 0.8, "reading": 0.2},
    "research": {"reading": 0.5, "evaluating": 0.5},
    "operations": {"planning": 0.7, "technical": 0.3},
    "sales": {"social": 0.6, "shopping": 0.4},
    "management": {"planning": 0.5, "social": 0.5},
}
# concepts of the education ecosystem (domains.concepts) whose mastery a skill draws on
EDU = {"data": ("mean", "percent", "frac_add", "frac_mul", "frac_simplify"),
       "software": ("binary", "power_mod", "lin_eq", "gcd"), "operations": ("mul", "div", "percent", "add", "sub"),
       "research": ("mean", "lin_eq")}

DEFAULT_DEMAND = {"per_week": [1.5, 3.0], "tasks": [1, 3], "hours": [16, 60], "value_cents_hour": [7_000, 13_000],
                  "standard": [0.5, 0.65], "deadline_days": [10, 21], "skills": None}
# firms of a population world whose configuration names none (civilisations): two contractors, one client
DEFAULT_FIRMS = [
    {"handle": "northwind", "name": "Northwind Digital", "goal": "profit", "skills": {"software": 0.6, "data": 0.5},
     "staff": 3, "headcount": 6, "salary": [300_000, 650_000]},
    {"handle": "bluefield", "name": "Bluefield Works", "goal": "growth",
     "skills": {"design": 0.5, "writing": 0.5, "operations": 0.5}, "staff": 3, "headcount": 7,
     "salary": [250_000, 550_000]},
    {"handle": "acme-buyer", "name": "Acme Holdings", "goal": "client", "staff": 0, "hire": False,
     "income_cents_month": 6_000_000, "demand": {}},
]


def _call(ctx: Ctx, who: str, tool: str, args: dict):
    """A tool call as `who`; an in-world refusal comes back as (None, error)."""
    try:
        return ctx.call(tool, args, as_principal=who), None
    except CrucibleError as e:
        return None, e


def declared_level(declared: list, skill: str) -> float:
    """What a firm infers about a worker's level in a skill from its public declaration (skills are listed
    strongest first; the first two are its strengths)."""
    if skill not in declared:
        return 0.25
    return 0.72 if declared.index(skill) < 2 else 0.45


def est_fit(declared: list, req: dict) -> float:
    """Estimated productivity on a task from declared skills (the labour market's skill-match form)."""
    if not req:
        return 1.0
    xs = [min(1.3, (declared_level(declared, k) + 0.05) / (v + 0.05)) for k, v in sorted(req.items())]
    return max(0.15, float(sum(xs) / len(xs)))


def staff_of(st, org: str) -> list[str]:
    """Employees of an organisation (simulated or not) under a live labour contract."""
    out = []
    for p in st.find_ids("labor_workers", "employer", org):
        w = st.tables["labor_workers"][p]
        c = st.get("contracts", w["contract"]) if w["contract"] else None
        if c is not None and c["status"] in ("active", "terminating"):
            out.append(p)
    return out


def load_hours(st, person: str) -> float:
    return float(sum(t["effort_hours"] * (1 - t["progress"]) for t in st.find("inst_tasks", "assignee", person)
                     if t["status"] == "assigned"))


def worker_profile(world, i: int) -> dict:
    """An individual's labour-market profile from its own population state (FR-FIRMS-JOBS): skill levels from its
    general skills (stretched: people differ more in occupational skills than in the composites), raised by its own
    education where the world has it (effective level-2 mastery of the concepts a skill draws on), declared skills
    (strongest first, certified ones first), interests, reservation wage from its income, hours, how much pay matters
    to it (its price sensitivity) and its patience (its loyalty)."""
    from .. import population as P
    from ..population.schema import PREF_NAMES, SKILL_NAMES
    f = P.features(world, np.array([int(i)]))
    sk = f["skills"][0]
    pr = f["prefs"][0]
    lv = {}
    for k, m in POP_TO_LABOR.items():
        raw = sum(w * float(sk[SKILL_NAMES.index(s)]) for s, w in m.items() if SKILL_NAMES.index(s) < sk.size)
        lv[k] = float(np.clip(0.5 + 1.8 * (raw - 0.5), 0.02, 0.95))
    certified = []
    learn = world.modules.get("learn")
    if learn is not None:
        know = learn.knowledge(world, int(i), world.now)
        certs = world.state.find("learn_certs", "person", ids.pid(int(i)))
        held = {c["concept"] for c in certs}
        for k, cs in EDU.items():
            m = [know[c][1] for c in cs if c in know]
            if m:
                lv[k] = float(np.clip(lv[k] + 0.35 * float(np.mean(m)) * (1.0 - lv[k]), 0.02, 0.97))
            if held & set(cs):
                certified.append(k)
    order = sorted(SKILLS, key=lambda k: (-(lv[k] + (0.5 if k in certified else 0.0)), k))
    inc = int(columns.col(world, "pop", "income_cents")[int(i)])
    emp = bool(f["employed"][0])
    # what it earns now (its income, more than an unemployed person's benefits) and what its two best skills are
    # worth on the market: skilled people ask more (adverse selection for an employer that pays little)
    top2 = float(np.mean(sorted(lv.values())[-2:]))
    own = inc * 30 * (1.0 if emp else 2.2)
    reservation = int(np.clip(0.4 * own + 0.6 * (150_000 + 750_000 * top2), 150_000, 1_500_000))
    raw = np.array([lv[k] for k in SKILLS]) + 0.3 * pr[PREF_NAMES.index("novelty_seeking")]
    interests = {k: round(float(x / raw.sum()), 4) for k, x in zip(SKILLS, raw)}
    return {"skills": {k: round(v, 3) for k, v in lv.items()}, "declared": order[:3], "certified": sorted(certified),
            "interests": interests, "reservation": reservation,
            "hours_week": int(30 + 15 * float(sk[SKILL_NAMES.index("planning")])),
            "wage_weight": round(float(0.35 + 0.5 * pr[PREF_NAMES.index("price_sensitivity")]), 3),
            "patience": round(float(0.25 + 0.2 * pr[PREF_NAMES.index("loyalty")]), 3)}


@register_module
class Firms(Module):
    name = "firms"
    host = "firms"
    depends = ("identity", "bank", "contracts", "business", "inst", "commit", "labor")

    def install(self, world):
        st = world.state
        st.declare_table("firms", index=("goal",))
        st.declare_table("firms_revenue", index=("firm",))
        st.declare_table("firms_decisions", index=("firm", "offer"))
        st.declare_table("firms_commissions", index=("client", "status"))
        world.hooks["firms"] = self
        for name, fn in (("on_commit_offer", self.on_commit_offer), ("on_inst_decision", self.on_inst_decision)):
            cur = world.hooks.get(name)

            def chained(*a, _inner=cur, _fn=fn):
                if _inner is not None:
                    _inner(*a)
                _fn(*a)
            world.hooks[name] = chained
        from ..substrate import bus
        for ev in ("labor.quit", "labor.left"):
            bus.subscribe(world, ev, self._on_attrition, owner="firms")
        self.has_pop = "pop" in world.modules
        if self.has_pop:
            from .. import population as P
            try:
                n = P.n_individuals(world)
            except NotFound:
                raise ProtocolError("firms: the population module 'pop' must come before 'firms'") from None
            columns.declare(world, JOBS, n, {"registered": ("uint8", (), 0), "searches": ("int32", (), 0),
                                             "applied": ("int32", (), 0), "last_search": ("int64", (), -1)})
            P.register_activity(world, "job_search", self.act_job_search, owner=self.name,
                                prior=float(self.config.get("job_search_prior", -0.2)), goals=(5,))

    # ------------------------------------------------------------------ genesis
    def genesis(self, ctx):
        if "firms" in self.config:
            specs = list(self.config.get("firms") or [])
        else:
            specs = [dict(f) for f in DEFAULT_FIRMS] if self.has_pop else []
        rng = ctx.world.rng_for("firms-genesis")
        for i, f in enumerate(specs):
            self.found(ctx, f, rng, i)
        for k, c in enumerate(self.config.get("commissions") or []):
            self.commission_to_seat(ctx, c, ctx.world.rng_for("firms-commission", k), k)
        if specs:
            ctx.schedule(ctx.now + 10 * 3600, "firms.day", {}, prio=74, period=DAY, key="firms.day")

    def found(self, ctx: Ctx, f: dict, rng, i: int) -> str:
        """Found a firm: its manager (an npc person with savings), the company and its account, its initial staff
        (hired at what each asked; from the population when the world has one), and its internal project."""
        idm = ident(ctx.world)
        bank = ctx.world.modules["bank"]
        handle = str(f["handle"])
        goal = str(f.get("goal", "profit"))
        if goal not in GOALS:
            raise ValueError(f"firms: unknown goal {goal!r}")
        unknown = sorted(k for k in (f.get("skills") or {}) if k not in SKILLS)
        if unknown:
            raise ValueError(f"firms: {handle}: unknown skills {unknown} (skills: {', '.join(SKILLS)})")
        cash = int(f.get("cash", 5_000_000))
        mgr = idm.create_person(ctx, f"{handle}-mgr", f"{f.get('name', handle)} manager", "npc",
                                {"role": "manager", "firm": f"o:{handle}"})
        macct = bank.open_account(ctx, mgr, "checking")
        bank.move(ctx, "acct_genesis", macct, cash + 200_000, "endowment", f"founding capital of {handle}")
        org = ctx.call("id.create_org", {"handle": handle, "name": str(f.get("name", handle)), "kind": "company"},
                       as_principal=mgr)["org"]
        ctx.call("biz.register", {"org": org, "account": macct, "sector": str(f.get("sector", "services"))},
                 as_principal=mgr)
        acct = ctx.call("bank.open_account", {"owner": org}, as_principal=mgr)["account"]
        ctx.call("bank.transfer", {"from_account": macct, "to": acct, "amount": cash, "memo": "capital"},
                 as_principal=mgr)
        skills = {k: float(v) for k, v in sorted((f.get("skills") or {}).items())}
        band = [int(x) for x in f.get("salary", [400_000, 800_000])]
        demand = None
        if goal == "client":
            demand = {**DEFAULT_DEMAND, **dict(f.get("demand") or {})}
        row = {"manager": mgr, "account": acct, "goal": goal, "skills": skills, "band": band,
               "headcount": int(f.get("headcount", f.get("staff", 0))), "margin": float(f.get("margin", MARGIN[goal])),
               "hire": bool(f.get("hire", True)), "contract": bool(f.get("contract", True)), "internal": None,
               "post": None, "post_t": None, "mission": sorted(f.get("mission") or []),
               "pledge_budget": int(f.get("pledge_cents", 0)), "pledged": 0, "drift": float(f.get("drift", 0.0)),
               "last_raise": 0, "attrition": 0, "competitors": sorted(f.get("competitors") or []),
               "budget": int(f.get("budget_cents", 0)), "thread": None, "slot": i, "demand": demand,
               "income": int(f.get("income_cents_month", 0)), "last_income": 0}
        ctx.emit("firms.firm", {"id": org, **row}, vis="internal")
        n_staff = int(f.get("staff", 0))
        lo, hi = f.get("staff_reservation", [band[0], band[1]])
        if n_staff and self.has_pop:
            self._staff_from_population(ctx, org, n_staff, list(skills) or list(SKILLS[:2]))
        else:
            for j in range(n_staff):
                self._initial_employee(ctx, org, f"{handle}-s{j}", rng, list(skills) or list(SKILLS[:2]),
                                       int(lo), int(hi), float(f.get("staff_strength", 0.7)))
        if skills and goal != "client":
            prj = ctx.call("inst.project_create", {"title": f"{handle} operations", "as_org": org},
                           as_principal=mgr)["project"]
            self._set(ctx, org, internal=prj)
        return org

    def register_worker(self, ctx, i: int) -> str:
        """Enter individual `i` into the labour market as a worker deciding for itself (labor, sim) with the
        profile of `worker_profile`; returns its principal."""
        p = ids.ensure_principal(ctx, int(i), account=True)
        if ctx.state.has("labor_workers", p):
            return p
        prof = worker_profile(ctx.world, int(i))
        ctx.world.modules["labor"].add_worker(ctx, p, prof["skills"], prof["interests"], prof["reservation"],
                                              sim=True, hours_week=prof["hours_week"],
                                              wage_weight=prof["wage_weight"], patience=prof["patience"],
                                              declared=prof["declared"])
        return p

    def _staff_from_population(self, ctx, org, n: int, firm_skills: list) -> None:
        """Initial staff: the individuals of the population whose own profiles fit the firm's skills best (among
        those not yet in the labour market), each hired at the salary it asks."""
        from .. import population as P
        w = ctx.world
        N = P.n_individuals(w)
        if not N:
            return
        reg = columns.col(w, JOBS, "registered")
        cand = [i for i in range(N) if not reg[i] and not ctx.state.has("labor_workers", ids.pid(i))]
        rng = w.rng_for("firms-staff", org)
        pool = sorted(rng.choice(cand, size=min(len(cand), 60), replace=False).tolist()) if cand else []
        scored = []
        for i in pool:
            prof = worker_profile(w, i)
            scored.append((-float(np.mean([prof["skills"][k] for k in firm_skills])), i))
        hired = []
        labor = w.modules["labor"]
        mgr = ctx.state.tables["firms"][org]["manager"]
        for _, i in sorted(scored)[:n]:
            p = self.register_worker(ctx, i)
            wk = ctx.state.tables["labor_workers"][p]
            ask = labor.ask(w, wk, org, {})
            oid = labor.make_offer(ctx, org, p, None, ask, "staff", 14, frm="worker")
            ctx.call("labor.respond", {"offer": oid, "action": "accept", "as_org": org}, as_principal=mgr)
            hired.append(i)
        if hired:
            columns.batch(ctx, JOBS, "firms.staff_init", np.array(hired, np.int64),
                          {"registered": np.ones(len(hired), np.uint8)}, extra={"org": org})

    def _initial_employee(self, ctx, org, handle, rng, firm_skills, lo, hi, strength) -> str:
        """A 4.0 simulated worker (worlds without the population only), employed at the salary it asks for."""
        idm = ident(ctx.world)
        bank = ctx.world.modules["bank"]
        labor = ctx.world.modules["labor"]
        pid = idm.create_person(ctx, handle, handle, "npc", {"role": "worker"})
        acct = bank.open_account(ctx, pid, "checking")
        bank.move(ctx, "acct_genesis", acct, 150_000, "endowment", f"savings of {pid}")
        k = 1 + int(rng.random() < 0.6)
        strong = list(rng.choice(sorted(firm_skills), size=min(k, len(firm_skills)), replace=False))
        skills = {}
        for s in SKILLS:
            if s in strong:
                skills[s] = round(float(np.clip(rng.normal(strength, 0.08), 0.45, 0.97)), 3)
            else:
                skills[s] = round(float(rng.uniform(0.05, 0.42)), 3)
        raw = rng.dirichlet(np.ones(len(SKILLS)) * 0.7) + np.array([0.15 if s in strong else 0.0 for s in SKILLS])
        raw = raw / raw.sum()
        interests = {s: round(float(raw[j]), 3) for j, s in enumerate(SKILLS)}
        top = sorted(SKILLS, key=lambda s: -skills[s])
        level = float(np.mean([skills[s] for s in top[:2]]))
        reservation = int(lo + (hi - lo) * _clip((level - 0.5) / 0.45) * float(rng.uniform(0.9, 1.1)))
        labor.add_worker(ctx, pid, skills, interests, reservation, sim=True, hours_week=int(rng.integers(34, 44)),
                         wage_weight=round(float(rng.uniform(0.4, 0.8)), 3),
                         patience=round(float(rng.uniform(0.3, 0.42)), 3), declared=top[:3])
        w = ctx.state.tables["labor_workers"][pid]
        ask = labor.ask(ctx.world, w, org, {})
        oid = labor.make_offer(ctx, org, pid, None, ask, "staff", 14, frm="worker")
        mgr = ctx.state.tables["firms"][org]["manager"]
        ctx.call("labor.respond", {"offer": oid, "action": "accept", "as_org": org}, as_principal=mgr)
        return pid

    # ------------------------------------------------------------------ a commission to a participant seat
    def commission_to_seat(self, ctx, c: dict, rng, k: int) -> None:
        """A client commissions a project from a participant seat's organisation (created for the seat, which owns
        it, with working capital): the project's task graph, its deadline, and an escrow offer paying each stage on
        measured delivery of the stage's final task at the quality bar."""
        from ._kit import make_brief
        st = ctx.state
        w = ctx.world
        idm = ident(w)
        bank = w.modules["bank"]
        seat = f"p:{c['seat']}"
        if not st.has("principals", seat):
            raise ProtocolError(f"firms: commission {k} names seat {seat}, which does not exist (module 'seat' must "
                                "come before 'firms')")
        oc = dict(c.get("org") or {})
        org = idm.create_org(ctx, str(oc.get("handle", f"{c['seat']}co")), str(oc.get("name", f"{c['seat']} & Co")),
                             seat, "company", {"seat": seat})
        acct = bank.open_account(ctx, org, "operating")
        cap = int(oc.get("capital_cents", 2_000_000))
        if cap:
            bank.move(ctx, "acct_genesis", acct, cap, "endowment", f"working capital of {org}")
        w.modules["business"].register_company(ctx, org, seat, "projects", fee=0, pay_from=None)
        client = f"o:{c['client']}"
        cf = st.get("firms", client)
        if cf is None:
            raise ProtocolError(f"firms: commission {k}: client {client} is not a firm of this world")
        brief = make_brief(rng, str(c.get("topology", "pipeline")), list(c.get("skills") or SKILLS[:4]),
                           levels=tuple(c.get("levels", (0.35, 0.6))), effort=tuple(c.get("effort", (24, 70))),
                           stages=int(c.get("stages", 3)), width=int(c.get("width", 6)), special=c.get("special"))
        days = int(c.get("deadline_days", 30))
        prj = ctx.call("inst.project_create", {"title": str(c.get("title", "Commissioned project")), "as_org": org,
                                               "lead": seat, "deadline_days": days, "partners": [client]},
                       as_principal=seat)["project"]
        ctx.call("inst.project_join", {"project": prj, "as_org": client, "accept": True}, as_principal=cf["manager"])
        key_to_task = {}
        for t in brief["tasks"]:
            r = ctx.call("inst.task_add", {"project": prj, "title": t["title"], "skills": t["skills"],
                                           "effort_hours": int(t["effort"]),
                                           "depends_on": [key_to_task[d] for d in t["deps"]]}, as_principal=seat)
            key_to_task[t["key"]] = r["task"]
        budget = int(c.get("budget_cents", 3_000_000))
        sinks = [key_to_task[x] for x in brief["stage_sinks"]]
        q = float(c.get("min_quality", 0.6))
        per = budget // max(1, len(sinks))
        ms = [{"amount_cents": per, "deliverable": {"kind": "task", "task": t, "min_quality": q},
               "due_in_s": int(days * DAY)} for t in sinks]
        terms = {"title": f"commission: {c.get('title', 'project')}", "payer": "$me", "milestones": ms,
                 "review_s": 2 * DAY, "dispute_s": 7 * DAY, "default_split": 0.5}
        r, err = _call(ctx, cf["manager"], "commit.offer", {"to": org, "terms": terms, "as_org": client,
                                                            "message": "We commission this project from you; each "
                                                            "stage is paid on delivery at the quality bar.",
                                                            "expires_in_s": 14 * DAY})
        cid = st.new_id("fcom")
        ctx.emit("firms.commission", {"id": cid, "client": client, "to": org, "seat": seat, "project": prj,
                                      "tasks": list(key_to_task.values()), "stage_tasks": sinks,
                                      "deadline": ctx.now + days * DAY, "budget": budget, "min_quality": q,
                                      "offer": (r or {}).get("offer"), "status": "offered", "kind": "seat",
                                      "error": None if err is None else str(err)[:200]}, vis=(client, org, seat))

    # ------------------------------------------------------------------ small helpers
    def _set(self, ctx, org, **changes) -> None:
        ctx.emit("firms.update", {"id": org, "changes": changes}, vis="internal")

    def firm(self, st, org) -> dict | None:
        return st.get("firms", org)

    def firm_of_manager(self, st, person) -> str | None:
        for org, f in st.scan("firms"):
            if f["manager"] == person:
                return org
        return None

    def cash(self, st, org) -> int:
        f = st.tables["firms"][org]
        a = st.get("accounts", f["account"])
        return int(a["balance"]) if a and a["status"] == "open" else 0

    def payroll(self, st, org) -> int:
        return sum(int(st.tables["labor_workers"][p]["salary"]) for p in staff_of(st, org))

    def hourly_cost(self, st, org) -> float:
        f = st.tables["firms"][org]
        people = staff_of(st, org)
        if not people:
            return (f["band"][0] + f["band"][1]) / 2 / 170.0
        sal = np.mean([st.tables["labor_workers"][p]["salary"] for p in people])
        hrs = np.mean([st.tables["labor_workers"][p]["hours_week"] for p in people]) * 4.33
        return float(sal / max(1.0, hrs))

    def public_market(self, st, declared: list) -> int:
        """The best salary in open postings whose required skills a worker declares (public information)."""
        best = 0
        for pid in st.find_ids("labor_posts", "status", "open"):
            p = st.tables["labor_posts"][pid]
            if p["skills"] and all(k in declared[:3] for k in p["skills"]):
                best = max(best, int(p["salary_max"]))
        return best

    def certified(self, st, worker: str) -> set:
        """Labour-market skills a worker holds a Board certificate for (public facts of the education domain)."""
        if "learn_certs" not in st.tables:
            return set()
        held = {c["concept"] for c in st.find("learn_certs", "person", worker)}
        return {k for k, cs in EDU.items() if held & set(cs)}

    def cert_seqs(self, st, worker: str, skills) -> list[int]:
        """The recorded certification events (`learn.certified`) behind a worker's certificates for the concepts the
        given labour-market skills draw on: the explicit causes (`refs`, crucible.civ.chains) of a decision that used
        those certificates."""
        if "learn_certs" not in st.tables:
            return []
        want = {c for k in skills for c in EDU.get(k, ())}
        return sorted({int(c["seq"]) for c in st.find("learn_certs", "person", worker)
                       if c["concept"] in want and "seq" in c})

    def _on_attrition(self, ctx, ev) -> None:
        org = ev.payload.get("org")
        f = ctx.state.get("firms", org)
        if f is None:
            return
        lo, hi = f["band"]
        self._set(ctx, org, attrition=f["attrition"] + 1, band=[int(lo * 1.04), int(hi * 1.06)])

    # ------------------------------------------------------------------ the daily round
    @process("day")
    def pr_day(self, ctx):
        for org in ctx.state.ids("firms"):
            err = contain(ctx, self._day, ctx, org)
            if err is not None:
                ctx.emit("firms.failed", {"id": org, "what": "day", "error": f"{type(err).__name__}: {err}"[:200]},
                         vis="internal")

    def _day(self, ctx, org):
        st = ctx.state
        f = st.tables["firms"][org]
        if st.tables["orgs"][org]["status"] != "active":
            return
        rng = ctx.rng("firm-day", org)
        if f["goal"] == "client":
            self._client_income(ctx, org, f)
            self._client_follow(ctx, org, f)
            self._client_demand(ctx, org, f, rng)
            return
        self._solvency(ctx, org, f)
        self._contract_work(ctx, org, f)
        self._internal_work(ctx, org, f, rng)
        if f["hire"]:
            self._recruit(ctx, org, f, rng)
        self._retain(ctx, org, f)

    def _solvency(self, ctx, org, f) -> None:
        st = ctx.state
        pay = self.payroll(st, org)
        if pay and self.cash(st, org) < pay:
            people = sorted(staff_of(st, org), key=lambda p: -(st.tables["labor_workers"][p]["joined"] or 0))
            if people:
                _call(ctx, f["manager"], "labor.dismiss", {"worker": people[0], "org": org})

    # ------------------------------------------------------------------ clients: finite, price-responsive demand
    def _client_income(self, ctx, org, f) -> None:
        """A client's bounded monthly income from its own customers outside the modelled world (its budget)."""
        if f["income"] <= 0 or ctx.now - int(f["last_income"]) < MONTH:
            return
        ctx.world.modules["bank"].move(ctx, "acct_external", f["account"], int(f["income"]), "income",
                                       f"{org} revenue from its own customers")
        self._set(ctx, org, last_income=ctx.now)

    def _client_demand(self, ctx, org, f, rng) -> None:
        """Commission a work package when the client needs one (a Poisson rate) and can afford it at its value."""
        st = ctx.state
        d = f["demand"] or {}
        lam = float(rng.uniform(*d["per_week"])) / 7.0
        if rng.random() > 1.0 - np.exp(-lam):
            return
        contractors = [o for o, g in st.scan("firms") if g["goal"] != "client" and g["contract"] and
                       st.tables["orgs"][o]["status"] == "active"]
        if not contractors:
            return
        skills = list(d.get("skills") or sorted({k for o in contractors for k in st.tables["firms"][o]["skills"]}))
        if not skills:
            return
        n_tasks = int(rng.integers(d["tasks"][0], d["tasks"][1] + 1))
        hours = [int(rng.integers(d["hours"][0], d["hours"][1] + 1)) for _ in range(n_tasks)]
        value_h = float(rng.uniform(*d["value_cents_hour"]))
        value = int(value_h * sum(hours))
        if self.cash(st, org) < value:
            return                                      # cannot afford it now: the need waits
        days = int(rng.integers(d["deadline_days"][0], d["deadline_days"][1] + 1))
        q = round(float(rng.uniform(*d["standard"])), 3)
        prj, err = _call(ctx, f["manager"], "inst.project_create", {"title": f"{org[2:]} work package",
                                                                     "as_org": org, "deadline_days": days})
        if err is not None:
            return
        prj = prj["project"]
        tasks = []
        for h in hours:
            pick = sorted(rng.choice(skills, size=min(len(skills), 1 + int(rng.random() < 0.4)), replace=False))
            req = {k: round(float(rng.uniform(0.3, 0.6)), 3) for k in pick}
            r, err = _call(ctx, f["manager"], "inst.task_add", {"project": prj, "title": "commissioned work",
                                                                 "effort_hours": h, "skills": req})
            if err is None:
                tasks.append(r["task"])
        if not tasks:
            return
        cid = st.new_id("fcom")
        ctx.emit("firms.commission", {"id": cid, "client": org, "to": None, "seat": None, "project": prj,
                                      "tasks": tasks, "stage_tasks": tasks, "deadline": ctx.now + days * DAY,
                                      "budget": value, "min_quality": q, "offer": None, "status": "open",
                                      "kind": "market", "asked": [], "error": None}, vis="public")
        self._ask_next(ctx, org, f, cid)

    def _ask_next(self, ctx, org, f, cid) -> None:
        """Offer the package to the contractor whose public staff fits best and has not been asked yet, opening at
        80% of the client's value (it accepts counters up to its value: `_consider_paying`)."""
        st = ctx.state
        c = st.tables["firms_commissions"][cid]
        tasks = [st.tables["inst_tasks"][t] for t in c["tasks"]]
        best = None
        for o, g in sorted(st.scan("firms")):
            if g["goal"] == "client" or not g["contract"] or o in c["asked"] or \
                    st.tables["orgs"][o]["status"] != "active":
                continue
            people = staff_of(st, o)
            if not people:
                continue
            fit = float(np.mean([max(est_fit(st.tables["labor_workers"][p]["declared"], t["skills"]) for p in people)
                                 for t in tasks]))
            key = (-fit, o)
            if best is None or key < best[0]:
                best = (key, o)
        if best is None:
            ctx.emit("firms.commission_status", {"id": cid, "status": "unfilled"}, vis="public")
            return
        to = best[1]
        due = int(max(DAY, c["deadline"] - ctx.now))
        per = int(0.8 * c["budget"] / len(tasks))
        ms = [{"amount_cents": per, "deliverable": {"kind": "task", "task": t, "min_quality": c["min_quality"]},
               "due_in_s": due} for t in c["tasks"]]
        terms = {"title": f"work package {cid}", "payer": "$me", "milestones": ms, "review_s": DAY,
                 "dispute_s": 5 * DAY, "default_split": 0.5}
        self._set(ctx, org, budget=int(c["budget"]))
        r, err = _call(ctx, f["manager"], "commit.offer", {"to": to, "terms": terms, "as_org": org,
                                                           "message": "Would you take this work package?",
                                                           "expires_in_s": 3 * DAY})
        ctx.emit("firms.commission_status", {"id": cid, "status": "asked", "to": to,
                                             "offer": (r or {}).get("offer")}, vis="public")

    def _client_follow(self, ctx, org, f) -> None:
        """Watch the client's open packages: accepted (a commitment exists) - done; refused or expired - ask the
        next contractor; past its deadline - give up."""
        st = ctx.state
        for cid in st.find_ids("firms_commissions", "client", org):
            c = st.tables["firms_commissions"][cid]
            if c["kind"] != "market" or c["status"] not in ("open", "asked"):
                continue
            if ctx.now >= c["deadline"]:
                ctx.emit("firms.commission_status", {"id": cid, "status": "lapsed"}, vis="public")
                continue
            if c["offer"] is None:
                continue
            thread = self._thread(st, c["offer"])
            if any(o["status"] == "accepted" for o in thread):
                acc = next(o for o in thread if o["status"] == "accepted")
                ctx.emit("firms.commission_status", {"id": cid, "status": "contracted", "to": c["to"],
                                                     "offer": acc["_id"]}, vis="public")
            elif all(o["status"] in ("rejected", "expired", "withdrawn", "countered") for o in thread) and \
                    not any(o["status"] == "open" for o in thread):
                self._ask_next(ctx, org, f, cid)

    def _thread(self, st, first: str) -> list[dict]:
        """Every offer of a negotiation thread (the first offer and the counters that follow it)."""
        return [{**dict(st.tables["commit_offers"][oid]), "_id": oid} for oid in st.find_ids("commit_offers", "thread",
                                                                                              first)]

    def _internal_work(self, ctx, org, f, rng) -> None:
        """Idle employees do internal operations tasks (they build a verified record; nobody pays for them)."""
        st = ctx.state
        if not f["internal"] or not f["skills"]:
            return
        people = staff_of(st, org)
        if not people:
            return
        open_internal = [tid for tid in st.find_ids("inst_tasks", "project", f["internal"])
                         if st.tables["inst_tasks"][tid]["status"] == "open"]
        idle = [p for p in people if self._idle(st, p, org)]
        while len(open_internal) < len(idle):
            ks = sorted(f["skills"])
            pick = sorted(rng.choice(ks, size=min(len(ks), 1 + int(rng.random() < 0.4)), replace=False))
            req = {k: round(float(f["skills"][k] * rng.uniform(0.5, 0.85)), 3) for k in pick}
            r, err = _call(ctx, f["manager"], "inst.task_add", {"project": f["internal"], "title": "operations work",
                                                                 "effort_hours": int(rng.integers(16, 40)),
                                                                 "skills": req})
            if err is not None:
                break
            open_internal.append(r["task"])
        for tid in open_internal:
            if not idle:
                break
            t = st.tables["inst_tasks"][tid]
            idle.sort(key=lambda p: (-est_fit(st.tables["labor_workers"][p]["declared"], t["skills"]), p))
            _call(ctx, f["manager"], "inst.task_assign", {"task": tid, "assignee": idle.pop(0)})

    def _idle(self, st, person, org) -> bool:
        """Free for internal work: nothing assigned outside the firm's own operations and under a week queued."""
        f = st.tables["firms"][org]
        for t in st.find("inst_tasks", "assignee", person):
            if t["status"] == "assigned" and t["project"] != f["internal"]:
                return False
        return load_hours(st, person) < 0.5 * st.tables["labor_workers"][person]["hours_week"]

    def _recruit(self, ctx, org, f, rng) -> None:
        st = ctx.state
        people = staff_of(st, org)
        pay = self.payroll(st, org)
        mid = (f["band"][0] + f["band"][1]) // 2
        affordable = self.cash(st, org) > 2 * (pay + mid)
        post = st.get("labor_posts", f["post"]) if f["post"] else None
        want = f["headcount"] - len(people)
        # a rival employer bids for the people its competitors hire: one more hire for each of a competitor's roles
        # (its postings, up to three) while the competitor employs anyone, posted for the latest such role's skills at
        # a salary well above the competitor's (a standing offer the competitor's people can leave for)
        bid = None
        if f["goal"] == "rival" and f["competitors"]:
            comp = [st.tables["labor_posts"][pid] for c in f["competitors"] for pid in st.find_ids("labor_posts", "org", c)]
            if comp and any(staff_of(st, c) for c in f["competitors"]):
                want += min(3, len(comp))
                last = max(comp, key=lambda q: (int(q["t"]) if "t" in q else 0, int(q["salary_max"]), str(q["title"])))
                bid = {"skills": {k: round(float(v), 3) for k, v in sorted(last["skills"].items())},
                       "salary_max": int(min(1.25 * f["band"][1], 1.3 * int(last["salary_max"])))}
        if want <= 0 or not affordable:
            if post is not None and post["status"] == "open":
                _call(ctx, f["manager"], "labor.close_post", {"post": f["post"]})
            return
        if post is None or post["status"] != "open":
            req = {k: round(min(0.6, v * 0.6), 3) for k, v in sorted(f["skills"].items())[:2]}
            top = int(f["band"][1])
            if bid is not None:
                req, top = bid["skills"], max(top, bid["salary_max"])
            r, err = _call(ctx, f["manager"], "labor.post", {"org": org, "title": f"{org[2:]} staff",
                                                             "skills": req, "salary_min": int(f["band"][0]),
                                                             "salary_max": int(top), "hours_week": 40})
            if err is None:
                self._set(ctx, org, post=r["post"], post_t=ctx.now)
            return
        # counters from applicants: accept within the band (a little above for a growth firm, up to its posted top
        # for a rival's bid), else counter once
        top = f["band"][1] * (1.08 if f["goal"] == "growth" else 1.0)
        if f["goal"] == "rival":
            top = max(top, int(post["salary_max"]))
        offers, _ = _call(ctx, f["manager"], "labor.offers", {"as_org": org})
        for o in offers or []:
            if o["from"] != "worker" or o["status"] != "open":
                continue
            if o["salary"] <= top:
                _call(ctx, f["manager"], "labor.respond", {"offer": o["offer"], "action": "accept", "as_org": org})
            elif o["round"] < 3:
                _call(ctx, f["manager"], "labor.respond", {"offer": o["offer"], "action": "counter",
                                                           "salary_cents_month": int(top), "as_org": org})
            else:
                _call(ctx, f["manager"], "labor.respond", {"offer": o["offer"], "action": "reject", "as_org": org})
        if ctx.now - int(f["post_t"] or 0) < 2 * DAY:
            return
        apps, err = _call(ctx, f["manager"], "labor.applicants", {"post": f["post"]})
        if err is not None:
            return
        pending = sum(1 for o in offers or [] if o["status"] == "open")
        cands = []
        held = {}
        for a in apps or []:
            if a["status"] != "applied" or a["profile"]["employer"] is not None:
                continue
            prof = a["profile"]
            fit = est_fit(prof["declared_skills"], {k: v * 0.8 for k, v in f["skills"].items()})
            rec = [v["mean_quality"] for v in prof["verified_record"].values()]
            cs = self.certified(st, a["worker"]) & set(f["skills"]) if "worker" in a else set()
            if cs:
                held[a["application"]] = (a["worker"], sorted(cs))
            cands.append((-(fit + (0.3 * float(np.mean(rec)) if rec else 0.0) + 0.15 * len(cs)), a["application"]))
        for _, app in sorted(cands)[:max(0, want - pending)]:
            hi = max(int(f["band"][1]), int(post["salary_max"])) if f["goal"] == "rival" else int(f["band"][1])
            sal = int(f["band"][0] + (hi - f["band"][0]) * float(rng.uniform(0.45, 0.7)))
            if app in held:
                # the shortlist used the candidate's Board certificates: name those education events as causes
                who, cs = held[app]
                ctx.emit("firms.shortlist", {"org": org, "application": app, "certified": cs,
                                             "refs": self.cert_seqs(st, who, cs)}, vis="internal")
            _call(ctx, f["manager"], "labor.offer", {"application": app, "salary_cents_month": sal})

    def _retain(self, ctx, org, f) -> None:
        st = ctx.state
        if ctx.now - int(f["last_raise"]) < MONTH:
            return
        ch = {"last_raise": ctx.now}
        if f["drift"]:
            ch["band"] = [int(f["band"][0] * (1 + f["drift"])), int(f["band"][1] * (1 + f["drift"]))]
        self._set(ctx, org, **ch)
        budget = self.cash(st, org) - 3 * self.payroll(st, org)
        for p in sorted(staff_of(st, org)):
            w = st.tables["labor_workers"][p]
            market = self.public_market(st, w["declared"])
            if market and w["salary"] < 0.95 * market and budget > 0:
                new = int(min(f["band"][1], market))
                if new > w["salary"]:
                    _call(ctx, f["manager"], "labor.adjust_pay", {"worker": p, "org": org, "salary_cents_month": new})
                    budget -= 3 * (new - w["salary"])

    # ------------------------------------------------------------------ contracted work
    def _contract_work(self, ctx, org, f) -> None:
        """Join the projects the firm has accepted work in, and staff the contracted tasks it may assign."""
        st = ctx.state
        for cid in st.find_ids("commitments", "payee", org):
            c = st.tables["commitments"][cid]
            if c["status"] not in ("active", "disputed"):
                continue
            for m in c["milestones"]:
                d = m["deliverable"]
                if d["kind"] != "task" or m["status"] != "pending":
                    continue
                t = st.get("inst_tasks", d["task"])
                if t is None:
                    continue
                self._join(ctx, org, f, t["project"])
                if t["status"] == "open" and self._may_lead(ctx, org, f, t["project"]):
                    people = [p for p in staff_of(st, org)
                              if load_hours(st, p) + t["effort_hours"] <= 3.5 * st.tables["labor_workers"][p]["hours_week"]]
                    if people:
                        people.sort(key=lambda p: (-est_fit(st.tables["labor_workers"][p]["declared"], t["skills"]),
                                                   load_hours(st, p), p))
                        _call(ctx, f["manager"], "inst.task_assign", {"task": d["task"], "assignee": people[0]})

    def _join(self, ctx, org, f, project) -> None:
        r = ctx.state.get("inst_partners", f"{project}|{org}")
        if r is not None and r["status"] == "invited":
            _call(ctx, f["manager"], "inst.project_join", {"project": project, "as_org": org, "accept": True})

    def _may_lead(self, ctx, org, f, project) -> bool:
        prj = ctx.state.get("inst_projects", project)
        if prj is None:
            return False
        return prj["lead"] == f["manager"] or ident(ctx.world).allowed(ctx, f["manager"], prj["owner"], "inst.lead",
                                                                        0, project)

    # ------------------------------------------------------------------ negotiation (commitments)
    def on_commit_offer(self, ctx, oid, to) -> None:
        f = ctx.state.get("firms", to)
        if f is None or not f["contract"]:
            return
        delay = int(ctx.world.rng_for("firm-consider", oid).integers(2, 12)) * 3600
        ctx.schedule(ctx.now + delay, "firms.consider", {"offer": oid}, prio=71)

    @process("consider")
    def pr_consider(self, ctx, offer):
        err = contain(ctx, self._consider, ctx, offer)
        if err is not None:
            ctx.emit("firms.failed", {"id": offer, "what": "consider", "error": f"{type(err).__name__}: {err}"[:200]},
                     vis="internal")

    def _record(self, ctx, org, oid, action, why, ask=None, offered=None) -> None:
        ctx.emit("firms.decision", {"id": f"{org}|{oid}", "firm": org, "offer": oid, "action": action,
                                    "why": why[:200], "ask": ask, "offered": offered}, vis="internal")

    def _respond(self, ctx, org, f, oid, action, why, terms=None, ask=None, offered=None) -> None:
        args = {"offer": oid, "action": action, "message": why, "as_org": org}
        if terms is not None:
            args["terms"] = terms
        _, err = _call(ctx, f["manager"], "commit.respond", args)
        self._record(ctx, org, oid, action if err is None else f"{action}-failed",
                     why if err is None else f"{why} ({err})", ask, offered)

    def _consider(self, ctx, oid):
        st = ctx.state
        o = st.get("commit_offers", oid)
        if o is None or o["status"] != "open" or ctx.now > o["expires"]:
            return
        org = o["to"]
        f = st.tables["firms"][org]
        c = o["concrete"]
        if c["payer"] == org:
            return self._consider_paying(ctx, org, f, o, oid)
        return self._consider_work(ctx, org, f, o, oid)

    def _competitor(self, f, other: str) -> bool:
        return other in f["competitors"]

    def plan_work(self, st, org, tasks_due: list[tuple[str, int]]) -> tuple[float, list[int], str | None]:
        """Estimated hours and the earliest due times (seconds from now) at which the firm can deliver each task
        with its staff's free hours; an error string when it cannot staff a task at all."""
        people = staff_of(st, org)
        if not people:
            return 0.0, [], "we have no staff"
        free = {p: load_hours(st, p) for p in people}
        total, dues = 0.0, []
        for tid, due_in in tasks_due:
            t = st.get("inst_tasks", tid)
            if t is None or t["status"] in ("done", "cancelled"):
                return 0.0, [], f"task {tid} is not open"
            best = max(people, key=lambda p: (est_fit(st.tables["labor_workers"][p]["declared"], t["skills"]), p))
            fit = est_fit(st.tables["labor_workers"][best]["declared"], t["skills"])
            if fit < 0.7:
                return 0.0, [], f"we lack staff for {', '.join(sorted(t['skills']))}"
            hours = t["effort_hours"] * (1 - t["progress"]) / fit
            daily = st.tables["labor_workers"][best]["hours_week"] / 7.0 * 0.85
            queue = free[best] + hours
            free[best] = queue
            total += hours
            dues.append(int((queue / daily + 2) * DAY))
        return total, dues, None

    def _consider_work(self, ctx, org, f, o, oid):
        st = ctx.state
        c = o["concrete"]
        payer = c["payer"]
        if f["goal"] == "client":
            return self._respond(ctx, org, f, oid, "reject", "we do not take contract work")
        kinds = {m["deliverable"]["kind"] for m in c["milestones"]}
        if kinds != {"task"}:
            return self._respond(ctx, org, f, oid, "reject", "we take work with measured task deliverables only")
        if f["goal"] in ("profit", "rival") and self._competitor(f, payer):
            return self._respond(ctx, org, f, oid, "reject", "we do not work for competitors")
        tasks_due = [(m["deliverable"]["task"], int(m["due_in_s"])) for m in c["milestones"]]
        hours, need, why = self.plan_work(st, org, tasks_due)
        if why:
            return self._respond(ctx, org, f, oid, "reject", why)
        offered = sum(m["amount"] for m in c["milestones"])
        margin = f["margin"]
        if f["goal"] == "mission":
            prj = {st.tables["inst_tasks"][t]["project"] for t, _ in tasks_due}
            aligned = any(st.tables["inst_projects"][p]["owner"] in f["mission"] or p in f["mission"] for p in prj)
            margin = 0.0 if aligned else 0.25
        ask = int(hours * self.hourly_cost(st, org) * (1 + margin))
        late = any(n > d for n, (_, d) in zip(need, tasks_due))
        if offered >= ask and not late:
            r, err = _call(ctx, f["manager"], "commit.respond", {"offer": oid, "action": "accept", "as_org": org})
            self._record(ctx, org, oid, "accept" if err is None else "accept-failed",
                         "terms meet our price and capacity" if err is None else str(err), ask, offered)
            if err is None:
                for t, _ in tasks_due:
                    self._join(ctx, org, f, st.tables["inst_tasks"][t]["project"])
            return
        if offered < 0.5 * ask or o["round"] >= 4:
            return self._respond(ctx, org, f, oid, "reject", f"far below our price ({ask} cents)", ask=ask,
                                 offered=offered)
        target = max(ask, int(offered * 1.02))
        scale = target / max(1, offered)
        ms = []
        for m, n, (_, d) in zip(c["milestones"], need, tasks_due):
            ms.append({"amount_cents": max(1, int(round(m["amount"] * scale))), "deliverable": m["deliverable"],
                       "due_in_s": int(max(d, n, 3600))})
        terms = {"title": c["title"], "payer": "$you", "milestones": ms, "review_s": c["review_s"],
                 "dispute_s": c["dispute_s"], "default_split": c["default_split"]}
        why = f"we can do it for {sum(x['amount_cents'] for x in ms)} cents" + (" with later due dates" if late else "")
        self._respond(ctx, org, f, oid, "counter", why, terms, ask, offered)

    def _consider_paying(self, ctx, org, f, o, oid):
        """Asked to pay: a client accepts counters within its budget for the work it commissioned; a mission firm
        pledges within its pledge budget, but only on deliverables the world checks (predicates)."""
        st = ctx.state
        c = o["concrete"]
        total = sum(m["amount"] for m in c["milestones"])
        if f["goal"] == "client":
            parent = st.get("commit_offers", o["parent"]) if o["parent"] else None
            budget = self._commission_budget(st, org, o)
            if parent is None or parent["from"] != org:
                return self._respond(ctx, org, f, oid, "reject", "we only negotiate the work we commissioned")
            mine = sorted(m["deliverable"].get("task", "") for m in parent["concrete"]["milestones"])
            theirs = sorted(m["deliverable"].get("task", "") for m in c["milestones"])
            if mine != theirs or any(m["deliverable"]["kind"] != "task" for m in c["milestones"]):
                return self._respond(ctx, org, f, oid, "reject", "the deliverables must stay the commissioned tasks")
            if any(m["deliverable"].get("min_quality", 0) < pm["deliverable"].get("min_quality", 0)
                   for m, pm in zip(sorted(c["milestones"], key=lambda m: m["deliverable"]["task"]),
                                    sorted(parent["concrete"]["milestones"], key=lambda m: m["deliverable"]["task"]))):
                return self._respond(ctx, org, f, oid, "reject", "the quality bar is not negotiable")
            prj = st.tables["inst_projects"][st.tables["inst_tasks"][mine[0]]["project"]]
            late = prj["deadline"] is not None and any(ctx.now + m["due_in_s"] > prj["deadline"] + DAY
                                                       for m in c["milestones"])
            if late:
                return self._respond(ctx, org, f, oid, "reject", "every milestone is due by the project deadline")
            if total <= budget:
                r, err = _call(ctx, f["manager"], "commit.respond", {"offer": oid, "action": "accept", "as_org": org})
                return self._record(ctx, org, oid, "accept" if err is None else "accept-failed",
                                    "within our budget" if err is None else str(err), budget, total)
            if o["round"] >= 3:
                return self._respond(ctx, org, f, oid, "reject", "we cannot pay that", ask=budget, offered=total)
            mid = int((total + sum(m["amount"] for m in parent["concrete"]["milestones"])) / 2)
            target = min(budget, mid)
            scale = target / max(1, total)
            ms = [{"amount_cents": max(1, int(m["amount"] * scale)), "deliverable": m["deliverable"],
                   "due_in_s": int(m["due_in_s"])} for m in c["milestones"]]
            terms = {"title": c["title"], "payer": "$me", "milestones": ms, "review_s": c["review_s"],
                     "dispute_s": c["dispute_s"], "default_split": c["default_split"]}
            return self._respond(ctx, org, f, oid, "counter", f"we can pay {sum(x['amount_cents'] for x in ms)}",
                                 terms, budget, total)
        if f["goal"] != "mission":
            return self._respond(ctx, org, f, oid, "reject", "we do not fund others' projects")
        if any(m["deliverable"]["kind"] != "predicate" for m in c["milestones"]):
            return self._respond(ctx, org, f, oid, "reject", "we pledge only against outcomes the world verifies")
        if not self._mission_predicates(st, f, c):
            return self._respond(ctx, org, f, oid, "reject", "this is outside our mission")
        room = f["pledge_budget"] - f["pledged"]
        if total <= room:
            r, err = _call(ctx, f["manager"], "commit.respond", {"offer": oid, "action": "accept", "as_org": org})
            if err is None:
                self._set(ctx, org, pledged=f["pledged"] + total)
            return self._record(ctx, org, oid, "accept" if err is None else "accept-failed",
                                "pledged" if err is None else str(err), room, total)
        if room <= 0 or o["round"] >= 3:
            return self._respond(ctx, org, f, oid, "reject", "our pledge budget is spent", ask=room, offered=total)
        scale = room / max(1, total)
        ms = [{"amount_cents": max(1, int(m["amount"] * scale)), "deliverable": m["deliverable"],
               "due_in_s": int(m["due_in_s"])} for m in c["milestones"]]
        terms = {"title": c["title"], "payer": "$me", "milestones": ms, "review_s": c["review_s"],
                 "dispute_s": c["dispute_s"], "default_split": c["default_split"]}
        self._respond(ctx, org, f, oid, "counter", f"we can pledge {sum(x['amount_cents'] for x in ms)}", terms, room,
                      total)

    def _commission_budget(self, st, org, o) -> int:
        """The value the client puts on the work package a counter-offer is about (its own commission record)."""
        root = o.get("thread") or o.get("parent")
        for cid in st.find_ids("firms_commissions", "client", org):
            c = st.tables["firms_commissions"][cid]
            if c["offer"] and (c["offer"] == root or c["offer"] == o.get("parent")):
                return int(c["budget"])
        return int(st.tables["firms"][org]["budget"])

    def _mission_predicates(self, st, f, c) -> bool:
        """A pledge is in the mission when every predicate reads the completion or delivery of a project the firm's
        mission names (or a project owned by an organisation it names)."""
        from ..goals.metrics import predicate_leaves
        for m in c["milestones"]:
            for lf in predicate_leaves(m["deliverable"]["predicate"]):
                if lf.get("metric") not in ("project_completion", "project_delivered", "project_quality"):
                    return False
                args = lf.get("args") or {}
                prj = st.get("inst_projects", args.get("project") or "")
                if prj is None or not (args.get("project") in f["mission"] or prj["owner"] in f["mission"]):
                    return False
        return True

    # ------------------------------------------------------------------ governance (votes)
    def on_inst_decision(self, ctx, did, voter) -> None:
        st = ctx.state
        org = voter if st.has("firms", voter) else self.firm_of_manager(st, voter)
        if org is None:
            return
        hours = 3 + int(ctx.world.rng_for("firm-vote-delay", did, voter).integers(0, 24))
        ctx.schedule(ctx.now + hours * 3600, "firms.vote", {"decision": did, "voter": voter}, prio=56)

    @process("vote")
    def pr_vote(self, ctx, decision, voter):
        err = contain(ctx, self._vote, ctx, decision, voter)
        if err is not None:
            ctx.emit("firms.failed", {"id": decision, "what": "vote", "error": f"{type(err).__name__}: {err}"[:200]},
                     vis="internal")

    def beneficiary(self, st, d) -> str | None:
        """Who a decision pays or contracts (an organisation or person), when it does."""
        a = d["action"]
        if a["kind"] != "tool":
            return None
        args = a.get("args") or {}
        to = args.get("to") or args.get("grantee") or args.get("payee")
        if isinstance(to, str) and to.startswith("acct_") and st.has("accounts", to):
            to = st.tables["accounts"][to]["owner"]
        if isinstance(to, str) and not to.startswith(("p:", "o:")) and st.find_ids("orgs", "handle", to):
            to = "o:" + to
        return to if isinstance(to, str) else None

    def vote_choice(self, ctx, org, d, rng) -> tuple[bool, str]:
        st = ctx.state
        f = st.tables["firms"][org]
        a = d["action"]
        to = self.beneficiary(st, d)
        mgr = f["manager"]
        proposer_org = d["by"]
        if a["kind"] == "tool" and to in (org, mgr):
            return True, "it pays us"
        if to is not None and self._competitor(f, to) and f["goal"] in ("profit", "rival", "growth"):
            return False, "it pays a competitor"
        if f["goal"] == "rival":
            return bool(rng.random() < 0.15), "we do not back the convener"
        if a["kind"] == "allocate" and a.get("task"):
            t = st.get("inst_tasks", a["task"]) or {}
            if t.get("assignee") in staff_of(st, org):
                return True, "the reward goes to our staff"
        if a["kind"] in ("dissolve", "expel") and f["goal"] != "rival":
            return False, "we oppose breaking up the organisation"
        if a["kind"] == "project_close":
            prj = st.get("inst_projects", a.get("project") or "")
            open_tasks = prj is not None and any(t["status"] not in ("done", "cancelled")
                                                 for t in st.find("inst_tasks", "project", a["project"]))
            return (not open_tasks), "the project is finished" if not open_tasks else "work is still open"
        p = VOTE_BASE[f["goal"]]
        if f["goal"] == "mission" and (d["org"] in f["mission"] or proposer_org in f["mission"]):
            p = 0.9
        return bool(rng.random() < p), "general support" if p >= 0.5 else "general reluctance"

    def _vote(self, ctx, did, voter):
        st = ctx.state
        d = st.get("inst_decisions", did)
        if d is None or d["status"] != "open" or st.has("inst_ballots", f"{did}|{voter}"):
            return
        org = voter if st.has("firms", voter) else self.firm_of_manager(st, voter)
        f = st.tables["firms"][org]
        yes, why = self.vote_choice(ctx, org, d, ctx.rng("firm-vote", did, voter))
        args = {"decision": did, "choice": "yes" if yes else "no"}
        if voter == org:
            args["as_org"] = org
        _, err = _call(ctx, f["manager"], "inst.vote", args)
        ctx.emit("firms.ballot", {"id": f"{did}|{voter}", "firm": org, "decision": did, "yes": bool(yes),
                                  "why": why, "ok": err is None}, vis="internal")

    # ------------------------------------------------------------------ the population's job search (activity)
    def act_job_search(self, ctx, idx, now):
        """A job-search session of each individual of the cohort (FR-FIRMS-JOBS): an individual not yet in the
        labour market enters it with its own profile; then, unless it is employed and content, it reads the open
        postings for itself and applies to the (at most two) that are best for it - pay against its reservation
        wage, how far it meets the requirements, its interest in the work, the employer's public record. The labour
        market decides the rest with it (offers, work, quitting). A worker already in the market first updates its
        profile with what it has learned since (`_education_update`). An individual whose labour-market record was
        entered by another path (not a simulated worker of this model, e.g. economy.jobs' own registration) is
        decided by that path: this session leaves it alone (one decision path per worker). Returns the session's
        utility."""
        from .. import population as P
        w = ctx.world
        st = ctx.state
        idx = np.asarray(idx, np.int64)
        util = np.zeros(idx.size)
        if idx.size == 0:
            return util
        labor = w.modules["labor"]
        reg = columns.col(w, JOBS, "registered")
        searches = columns.col(w, JOBS, "searches")
        applied = columns.col(w, JOBS, "applied")
        new_reg, new_s, new_a, last = [], [], [], []
        posts = None
        n_edu = n_other = 0
        last_s = columns.col(w, JOBS, "last_search")
        for j, i in enumerate(idx.tolist()):
            known = bool(reg[i])
            p = self.register_worker(ctx, i)
            wk = st.tables["labor_workers"][p]
            if not wk["sim"]:
                # in the market through another path that decides for it: nothing to do here
                n_other += 1
                new_reg.append(int(reg[i]))
                new_s.append(int(searches[i]))
                new_a.append(int(applied[i]))
                last.append(int(last_s[i]))
                continue
            if known and "learn" in w.modules:
                wk, upd = self._education_update(ctx, p, i, wk)
                n_edu += int(upd)
            n_app = 0
            content = wk["employer"] is not None and wk["satisfaction"] >= wk["patience"] + 0.1
            if content:
                util[j] = 0.2 + 0.6 * float(wk["satisfaction"])
            else:
                if posts is None:
                    r, err = _call(ctx, p, "labor.postings", {"limit": 100})
                    posts = r if err is None and isinstance(r, list) else []
                opts = []
                for po in posts:
                    if po["org"] in wk["blacklist"] or po["org"] == wk["employer"] or \
                            st.has("labor_apps", f"{po['post']}|{p}"):
                        continue
                    req = po.get("skills") or {}
                    gap = sum(max(0.0, v - wk["skills"].get(k, 0.0)) for k, v in req.items())
                    if gap > 0.25:
                        continue
                    risk = labor.employer_risk(w, po["org"])
                    if po["salary_max"] < 0.85 * wk["reservation"] * risk:
                        continue
                    u = wk["wage_weight"] * po["salary_max"] / max(1, wk["reservation"]) + \
                        (1 - wk["wage_weight"]) * labor.interest_fit(wk, req) - 2.0 * gap
                    opts.append((-u, po["post"]))
                for _, post in sorted(opts)[:2]:
                    _, err = _call(ctx, p, "labor.apply", {"post": post, "note": "I would like this job"})
                    n_app += int(err is None)
                util[j] = (0.1 if n_app else -0.3) if wk["employer"] is None else (0.15 if n_app else 0.0)
            new_reg.append(1)
            new_s.append(int(searches[i]) + 1)
            new_a.append(int(applied[i]) + n_app)
            last.append(int(now))
        P.record_decisions(ctx, "firms.jobsearch", idx, {
            "registered": np.array(new_reg, np.uint8), "searches": np.array(new_s, np.int32),
            "applied": np.array(new_a, np.int32), "last_search": np.array(last, np.int64)}, table=JOBS, **JOB_MODEL,
            extra={"applications": int(sum(new_a) - int(applied[idx].sum())), "education_updates": n_edu,
                   "other_path": n_other})
        del reg, last_s
        return util

    def _education_update(self, ctx, p: str, i: int, wk: dict) -> tuple[dict, bool]:
        """What a worker learned since it entered the labour market (FR-FIRMS-JOBS): its occupational skills rise to
        what its current mastery supports (`worker_profile`; they never fall here - forgetting of occupational skill
        is not modelled) and its declared skills follow its profile when its strengths or certificates changed (it
        updates its CV). Education thereby changes whom employers shortlist and the real quality of its work."""
        prof = worker_profile(ctx.world, int(i))
        cur = {k: float(wk["skills"].get(k, 0.0)) for k in SKILLS}
        sk = {k: round(max(cur[k], float(prof["skills"][k])), 3) for k in SKILLS}
        ch = {}
        if any(sk[k] >= cur[k] + 0.02 for k in SKILLS):
            ch["skills"] = sk
        dec = list(prof["declared"])
        new_cert = [k for k in prof["certified"] if k not in wk["declared"]]
        if set(dec[:2]) != set(wk["declared"][:2]) or new_cert:
            ch["declared"] = dec
        if not ch:
            return wk, False
        # provenance (explicit `refs`, crucible.civ.chains): the recorded education events this update rests on - the
        # individual's latest study session when a skill education draws on rose, the certificates it newly declares
        st = ctx.state
        refs = set(self.cert_seqs(st, p, new_cert)) if new_cert else set()
        rose = [k for k in EDU if k in sk and sk[k] >= cur[k] + 0.02]
        if rose and "learn_sessions" in st.tables:
            ss = [int(r["seq"]) for r in st.find("learn_sessions", "i", int(i)) if "seq" in r]
            if ss:
                refs.add(max(ss))
        if refs:
            ctx.emit("firms.reprofile", {"person": p, "changed": sorted(ch), "rose": rose, "certified": new_cert,
                                         "refs": sorted(refs)}, vis="internal")
        ctx.world.modules["labor"]._upd(ctx, p, **ch)
        return {**wk, **ch}, True

    # ------------------------------------------------------------------ appliers
    @applier("firms.firm")
    def ap_firm(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("firms", p.pop("id"), p | {"t": ev.t})

    @applier("firms.update")
    def ap_update(self, ev):
        self.world.state.patch("firms", ev.payload["id"], **ev.payload["changes"])

    @applier("firms.commission")
    def ap_commission(self, ev):
        p = dict(ev.payload)
        p.setdefault("asked", [])
        self.world.state.insert("firms_commissions", p.pop("id"), p | {"t": ev.t})

    @applier("firms.commission_status")
    def ap_commission_status(self, ev):
        p = ev.payload
        st = self.world.state
        c = st.get("firms_commissions", p["id"])
        if c is None:
            return
        ch = {"status": p["status"]}
        if p.get("to"):
            ch["to"] = p["to"]
            if p["to"] not in c["asked"]:
                ch["asked"] = list(c["asked"]) + [p["to"]]
        if "offer" in p:
            ch["offer"] = p["offer"]
        st.patch("firms_commissions", p["id"], **ch)

    @applier("firms.decision")
    def ap_decision(self, ev):
        p = ev.payload
        self.world.state.put("firms_decisions", p["id"], {k: p[k] for k in ("firm", "offer", "action", "why", "ask",
                                                                            "offered")} | {"t": ev.t})

    @applier("firms.ballot")
    def ap_ballot(self, ev):
        return None

    @applier("firms.reprofile")
    def ap_reprofile(self, ev):
        return None          # a record of the cause of a profile update (labor.worker_update writes the profile)

    @applier("firms.shortlist")
    def ap_shortlist(self, ev):
        return None          # a record of the certificates an offer decision used (labor.offer writes the offer)

    @applier("firms.failed")
    def ap_failed(self, ev):
        return None

    def check_invariants(self, world):
        st = world.state
        errs = []
        for org, f in st.scan("firms"):
            if f["goal"] not in GOALS:
                errs.append(f"{org}: unknown goal")
            if f["pledged"] > f["pledge_budget"]:
                errs.append(f"{org}: pledged {f['pledged']} beyond its budget {f['pledge_budget']}")
        if self.has_pop and JOBS in columns.tables(world):
            reg = columns.col(world, JOBS, "registered")
            for i in np.flatnonzero(reg).tolist()[:2000]:
                if not st.has("labor_workers", ids.pid(i)):
                    errs.append(f"individual {i} registered for jobs without a labour-market record")
                    break
        return errs
