"""Autonomous organisations for the Tier 2 domains (module `firms`, host `firms`; MASTER_SPEC Parts 6.6, 9.9, 9.12).

A firm is an identity organisation registered as a company, run by a manager (a person of kind `npc`) who acts only
through ordinary tools as itself or for its organisation, from the firm's own books and what is public. Firms are
the independent organisations of the labour-marketplace and organisational-ecosystem domains: rival employers,
clients who commission work, contractors who take it, partners in shared projects and members of consortia.

Each firm has a **goal** that shapes every decision it takes - `profit` (prices work at a high margin, votes for
what pays it), `growth` (takes work at a thin margin when it has capacity, keeps hiring), `mission` (works at cost
for projects in its mission and pledges money conditionally on the shared outcome), `rival` (competes with the
participant: prices its work high and votes against its proposals) or `client` (commissions a project and pays on
measured delivery). What a firm does, by mechanism:

* **Staff and recruitment** (`labor`): initial employees are simulated workers strong in the firm's skills, hired
  at the salary they asked for (an employment contract each). A firm below its headcount posts jobs within its
  salary band, offers to the applicants whose declared skills and verified record fit best, and answers counters
  up to the top of its band. It stops hiring and dismisses its newest employee when payroll outruns its cash.
* **Internal work and its market**: a firm keeps an internal project with a stream of tasks in its skills and
  assigns them to idle employees; finished work is sold to the outside economy (rest-of-world account) at the
  firm's hourly rate times the measured quality. This keeps employees busy, so contract work competes for their
  hours, and pays the firm's salaries.
* **Retention**: each quit raises the firm's salary band; once a month it offers raises to employees paid below the
  best public posting for their declared skills (a firm with `drift` also raises its band every month).
* **Contracting and negotiation** (`commit`): offered a commitment to do work (task deliverables), the manager
  estimates the hours from its staff's declared skills, checks its capacity before each due date, prices the work
  at its hourly cost plus its goal's margin, and accepts, counters (price and, if needed, due dates) or rejects.
  Asked to pay (a pledge), a mission firm accepts conditional deliverables (predicates on the shared outcome)
  within its pledge budget; a client accepts counters up to its budget. Accepted work: the firm joins the project
  it was invited to and, if the lead delegated `inst.lead` to it, staffs the task with its best-fitting free
  employee (otherwise the lead assigns its employees).
* **Governance** (`inst`): a firm (venture partner) or its manager (member of a consortium) votes on decisions
  from its goal: for what pays it or advances its mission, against payments to its competitors and, as a rival,
  against the participant.

Hidden from every participant: goals, margins, budgets and the band (they are the firm's private books); what
others see is the firm's public conduct (postings, offers, counters, votes, contracts, breaches).
"""
from __future__ import annotations

import numpy as np

from ..commerce.labor import SKILLS, _clip
from ..core.errors import CrucibleError, contain
from ..core.kernel import Ctx, Module, applier, process, register_module
from ..services.identity import ident

DAY = 86400
MONTH = 30 * DAY
GOALS = ("profit", "growth", "mission", "rival", "client")
MARGIN = {"profit": 0.35, "growth": 0.12, "mission": 0.0, "rival": 0.9, "client": 0.3}
VOTE_BASE = {"profit": 0.55, "growth": 0.6, "mission": 0.75, "rival": 0.25, "client": 0.5}


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


@register_module
class Firms(Module):
    name = "firms"
    host = "firms"
    depends = ("identity", "bank", "contracts", "business", "inst", "commit", "labor")

    def install(self, world):
        st = world.state
        st.declare_table("firms", index=("goal",))
        st.declare_table("firms_sales", index=("firm",))
        st.declare_table("firms_decisions", index=("firm", "offer"))
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

    # ------------------------------------------------------------------ genesis
    def genesis(self, ctx):
        specs = list(self.config.get("firms") or [])
        rng = ctx.world.rng_for("firms-genesis")
        for i, f in enumerate(specs):
            self.found(ctx, f, rng, i)
        if specs:
            ctx.schedule(ctx.now + 10 * 3600, "firms.day", {}, prio=74, period=DAY, key="firms.day")

    def found(self, ctx: Ctx, f: dict, rng, i: int) -> str:
        """Found a firm: its manager (an npc person with savings), the company and its account, its initial staff
        (hired at what each asked), and its internal project."""
        idm = ident(ctx.world)
        bank = ctx.world.modules["bank"]
        handle = str(f["handle"])
        goal = str(f.get("goal", "profit"))
        if goal not in GOALS:
            raise ValueError(f"firms: unknown goal {goal!r}")
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
        skills = {k: float(v) for k, v in sorted((f.get("skills") or {}).items()) if k in SKILLS}
        band = [int(x) for x in f.get("salary", [400_000, 800_000])]
        row = {"manager": mgr, "account": acct, "goal": goal, "skills": skills, "band": band,
               "headcount": int(f.get("headcount", f.get("staff", 0))), "margin": float(f.get("margin", MARGIN[goal])),
               "rate": int(f.get("rate_cents_hour", 0)), "hire": bool(f.get("hire", True)),
               "contract": bool(f.get("contract", True)), "internal": None, "post": None, "post_t": None,
               "mission": sorted(f.get("mission") or []), "pledge_budget": int(f.get("pledge_cents", 0)),
               "pledged": 0, "drift": float(f.get("drift", 0.0)), "last_raise": 0, "attrition": 0,
               "competitors": sorted(f.get("competitors") or []), "budget": int(f.get("budget_cents", 0)),
               "thread": None, "slot": i}
        ctx.emit("firms.firm", {"id": org, **row}, vis="internal")
        n_staff = int(f.get("staff", 0))
        lo, hi = f.get("staff_reservation", [band[0], band[1]])
        for j in range(n_staff):
            self._initial_employee(ctx, org, f"{handle}-s{j}", rng, list(skills) or list(SKILLS[:2]),
                                   int(lo), int(hi), float(f.get("staff_strength", 0.7)))
        if row["rate"] > 0:
            prj = ctx.call("inst.project_create", {"title": f"{handle} operations", "as_org": org},
                           as_principal=mgr)["project"]
            self._set(ctx, org, internal=prj)
        return org

    def _initial_employee(self, ctx, org, handle, rng, firm_skills, lo, hi, strength) -> str:
        """A simulated worker strong in one or two of the firm's skills, employed at the salary it asks for."""
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
        self._solvency(ctx, org, f)
        self._sell_output(ctx, org, f)
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

    def _sell_output(self, ctx, org, f) -> None:
        """Finished internal work is sold to the outside economy at the firm's rate times measured quality."""
        st = ctx.state
        if not f["internal"] or f["rate"] <= 0:
            return
        bank = ctx.world.modules["bank"]
        for tid in st.find_ids("inst_tasks", "project", f["internal"]):
            t = st.tables["inst_tasks"][tid]
            if t["status"] != "done" or st.has("firms_sales", tid):
                continue
            q = float(t["quality"] or 0.0)
            amount = int(t["effort_hours"] * f["rate"] * q)
            if amount > 0:
                bank.move(ctx, "acct_external", f["account"], amount, "external_sale",
                          f"{org} output of task {tid}", ref=tid)
            ctx.emit("firms.sale", {"id": tid, "firm": org, "amount": amount, "quality": round(q, 4)},
                     vis=(org,))

    def _internal_work(self, ctx, org, f, rng) -> None:
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
        if want <= 0 or not affordable:
            if post is not None and post["status"] == "open":
                _call(ctx, f["manager"], "labor.close_post", {"post": f["post"]})
            return
        if post is None or post["status"] != "open":
            req = {k: round(min(0.6, v * 0.6), 3) for k, v in sorted(f["skills"].items())[:2]}
            r, err = _call(ctx, f["manager"], "labor.post", {"org": org, "title": f"{org[2:]} staff",
                                                             "skills": req, "salary_min": int(f["band"][0]),
                                                             "salary_max": int(f["band"][1]), "hours_week": 40})
            if err is None:
                self._set(ctx, org, post=r["post"], post_t=ctx.now)
            return
        # counters from applicants: accept within the band (a little above for a growth firm), else counter once
        top = f["band"][1] * (1.08 if f["goal"] == "growth" else 1.0)
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
        for a in apps or []:
            if a["status"] != "applied" or a["profile"]["employer"] is not None:
                continue
            prof = a["profile"]
            fit = est_fit(prof["declared_skills"], {k: v * 0.8 for k, v in f["skills"].items()})
            rec = [v["mean_quality"] for v in prof["verified_record"].values()]
            cands.append((-(fit + (0.3 * float(np.mean(rec)) if rec else 0.0)), a["application"]))
        for _, app in sorted(cands)[:max(0, want - pending)]:
            sal = int(f["band"][0] + (f["band"][1] - f["band"][0]) * float(rng.uniform(0.45, 0.7)))
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
            if total <= f["budget"]:
                r, err = _call(ctx, f["manager"], "commit.respond", {"offer": oid, "action": "accept", "as_org": org})
                return self._record(ctx, org, oid, "accept" if err is None else "accept-failed",
                                    "within our budget" if err is None else str(err), f["budget"], total)
            if o["round"] >= 3:
                return self._respond(ctx, org, f, oid, "reject", "we cannot pay that", ask=f["budget"], offered=total)
            mid = int((total + sum(m["amount"] for m in parent["concrete"]["milestones"])) / 2)
            target = min(f["budget"], mid)
            scale = target / max(1, total)
            ms = [{"amount_cents": max(1, int(m["amount"] * scale)), "deliverable": m["deliverable"],
                   "due_in_s": int(m["due_in_s"])} for m in c["milestones"]]
            terms = {"title": c["title"], "payer": "$me", "milestones": ms, "review_s": c["review_s"],
                     "dispute_s": c["dispute_s"], "default_split": c["default_split"]}
            return self._respond(ctx, org, f, oid, "counter", f"we can pay {sum(x['amount_cents'] for x in ms)}",
                                 terms, f["budget"], total)
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

    # ------------------------------------------------------------------ appliers
    @applier("firms.firm")
    def ap_firm(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("firms", p.pop("id"), p | {"t": ev.t})

    @applier("firms.update")
    def ap_update(self, ev):
        self.world.state.patch("firms", ev.payload["id"], **ev.payload["changes"])

    @applier("firms.sale")
    def ap_sale(self, ev):
        p = ev.payload
        self.world.state.insert("firms_sales", p["id"], {"firm": p["firm"], "amount": p["amount"],
                                                         "quality": p["quality"], "t": ev.t})

    @applier("firms.decision")
    def ap_decision(self, ev):
        p = ev.payload
        self.world.state.put("firms_decisions", p["id"], {k: p[k] for k in ("firm", "offer", "action", "why", "ask",
                                                                            "offered")} | {"t": ev.t})

    @applier("firms.ballot")
    def ap_ballot(self, ev):
        return None

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
        return errs
