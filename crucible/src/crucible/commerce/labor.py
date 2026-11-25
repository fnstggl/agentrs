"""Labour marketplace (module `labor`, host `labor`; MASTER_SPEC Part 9.9).

Workers are persons with skills, preferences and a reservation wage. Simulated workers (kind `npc`) act only as
their own principal through ordinary tools, from their own state and what is public: open postings, the employer's
public record (breached payment contracts, breached commitments, company age), their own pay history and workload.
Their skills are private: an employer sees what a worker declares (skill names, no levels) and the *verified
record* the world keeps of the work the worker has actually done (tasks completed per skill and their measured
quality). Agents can be workers too (`labor.register`) and do the work themselves.

* **Recruitment**: postings with required skill levels and a salary range (`labor.post`), applications
  (`labor.apply`, or the simulated job search), applicants with their public profiles (`labor.applicants`).
* **Compensation and negotiation**: an offer (`labor.offer`) is a salary, a title and a notice period; the worker
  accepts, rejects or counters (`labor.respond`), and the employer may counter back. Acceptance creates a 4.0
  employment contract (monthly salary payment + membership + notice) signed by both sides, so unpaid salary is a
  contract breach like any other. Pay changes (`labor.adjust_pay`) are offers too.
* **Delegation and project dependencies**: work is `inst` project tasks (required skills, effort, dependencies),
  assigned by the project lead or anyone the owner delegated `inst.lead` to (`id.grant`). A task can be worked only
  when the tasks it depends on are done. A simulated worker refuses an assignment it cannot fit in its hours.
* **Performance**: each day an employed simulated worker works its hours (scaled by effort, which falls with
  fatigue and dissatisfaction) on its assigned, unblocked tasks; progress depends on the match between its skills
  and the task's requirements; a finished task's quality is measured from that match (seeded per task) and recorded
  in the worker's verified record; hours worked are recorded as labour contributions to the employer.
* **Retention and independent agency**: satisfaction follows pay relative to what the market offers for the
  worker's skills, interest in the work, fatigue, and unpaid wages; dissatisfied workers with a better outside
  option (or very dissatisfied ones) quit by terminating their contract with its notice. Workers whose employer
  breaches their contract leave and will not work for it again.

Evaluator metrics (goals/metrics.py, subject = the organisation): `labor_employees`, `labor_retention`,
`project_completion`, `project_quality`, `project_delivered`.
"""
from __future__ import annotations

import numbers

import numpy as np

from ..core import schema as S
from ..core.errors import Conflict, CrucibleError, Forbidden, InvalidAction, NotFound, contain
from ..core.kernel import Ctx, Module, applier, endpoint, process, register_module
from ..services.identity import ident

DAY = 86400
MONTH = 30 * DAY
SKILLS = ("software", "data", "design", "writing", "research", "operations", "sales", "management")


def _clip(x, lo=0.0, hi=1.0) -> float:
    return float(min(hi, max(lo, x)))


def skill_match(skills: dict, req: dict) -> float:
    """Productivity of a worker on a task: 1.0 when every required skill is met, less when short (never below
    0.15), up to 1.3 when well above the requirements."""
    if not req:
        return 1.0
    xs = [min(1.3, (skills.get(k, 0.0) + 0.05) / (v + 0.05)) for k, v in sorted(req.items())]
    return max(0.15, float(sum(xs) / len(xs)))


def task_quality(skills: dict, req: dict, noise: float) -> float:
    if not req:
        base = 0.6 + 0.3 * float(np.mean([skills.get(k, 0.0) for k in SKILLS]))
    else:
        base = 0.65 + float(np.mean([skills.get(k, 0.0) - v for k, v in sorted(req.items())]))
    return _clip(base + noise)


@register_module
class Labor(Module):
    name = "labor"
    depends = ("identity", "bank", "contracts", "inst")

    def install(self, world):
        st = world.state
        st.declare_table("labor_workers", index=("employer", "sim"))
        st.declare_table("labor_posts", index=("org", "status"))
        st.declare_table("labor_apps", index=("post", "worker", "status"))
        st.declare_table("labor_offers", index=("worker", "org", "status", "thread"))
        st.declare_table("labor_perf", index=("worker", "org"))
        world.hooks["labor"] = self
        # notification hooks are chained (modules installed later chain on top of these in turn)
        for name, fn in (("on_contract_ended", self.on_contract_ended), ("on_task_assigned", self.on_task_assigned),
                         ("on_inst_decision", self.on_inst_decision)):
            cur = world.hooks.get(name)

            def chained(*a, _inner=cur, _fn=fn):
                if _inner is not None:
                    _inner(*a)
                _fn(*a)
            world.hooks[name] = chained

    def cfg(self, k, d):
        return self.config.get(k, d)

    def genesis(self, ctx):
        ctx.schedule(ctx.now + 9 * 3600, "labor.day", {}, prio=73, period=DAY, key="labor.day")
        n = int(self.cfg("workers", 0))
        if n:
            rng = ctx.world.rng_for("labor-genesis")
            lo, hi = self.cfg("reservation_range", [300_000, 900_000])
            for i in range(n):
                self.create_sim_worker(ctx, f"{self.cfg('prefix', 'worker')}{i}", rng, int(lo), int(hi))

    # ================================================================== workers
    def create_sim_worker(self, ctx, handle: str, rng, lo: int, hi: int) -> str:
        """A simulated worker: a person (kind npc) with a checking account, private skills (two strengths, the
        rest weak), interests over kinds of work, hours, a reservation wage that rises with skill, and patience."""
        st = ctx.state
        idm = ident(ctx.world)
        pid = idm.create_person(ctx, handle, handle, "npc", {"role": "worker"})
        bank = ctx.world.modules["bank"]
        acct = bank.open_account(ctx, pid, "checking")
        savings = int(self.cfg("savings_cents", 200_000))
        if savings:
            bank.move(ctx, "acct_genesis", acct, savings, "endowment", f"savings of {pid}")
        strong = rng.choice(len(SKILLS), size=2, replace=False)
        skills = {}
        for j, k in enumerate(SKILLS):
            lvl = rng.uniform(0.55, 0.95) if j in strong else rng.uniform(0.05, 0.45)
            skills[k] = round(float(lvl), 3)
        raw = rng.dirichlet(np.ones(len(SKILLS)) * 0.7) + np.array([0.15 if j in strong else 0.0
                                                                    for j in range(len(SKILLS))])
        raw = raw / raw.sum()
        interests = {k: round(float(raw[j]), 3) for j, k in enumerate(SKILLS)}
        top = sorted(SKILLS, key=lambda k: -skills[k])
        level = float(np.mean([skills[k] for k in top[:2]]))
        reservation = int(lo + (hi - lo) * _clip((level - 0.55) / 0.4) * float(rng.uniform(0.85, 1.15)))
        self.add_worker(ctx, pid, skills, interests, reservation, sim=True,
                        hours_week=int(rng.integers(30, 46)), wage_weight=round(float(rng.uniform(0.4, 0.85)), 3),
                        patience=round(float(rng.uniform(0.3, 0.45)), 3), declared=top[:3])
        return pid

    def add_worker(self, ctx, person: str, skills: dict, interests: dict, reservation: int, *, sim: bool,
                   hours_week: int = 40, wage_weight: float = 0.6, patience: float = 0.35,
                   declared: list | None = None) -> None:
        sk = {k: round(float(_clip(skills.get(k, 0.0))), 3) for k in SKILLS}
        it = {k: round(float(_clip(interests.get(k, 0.0))), 3) for k in SKILLS}
        ctx.emit("labor.worker", {"person": person, "sim": bool(sim), "skills": sk, "interests": it,
                                  "reservation": int(reservation), "hours_week": int(hours_week),
                                  "wage_weight": float(wage_weight), "patience": float(patience),
                                  "declared": [d for d in (declared or []) if d in SKILLS][:5]},
                 vis="internal" if sim else (person,))

    def _upd(self, ctx, person, **changes) -> None:
        ctx.emit("labor.worker_update", {"person": person, "changes": changes}, vis="internal")

    def market_wage(self, st, w) -> int | None:
        """The best salary on offer in open postings the worker qualifies for (public information)."""
        best = None
        for pid in st.find_ids("labor_posts", "status", "open"):
            p = st.tables["labor_posts"][pid]
            if p["org"] in w["blacklist"]:
                continue
            if all(w["skills"].get(k, 0.0) >= 0.9 * v for k, v in p["skills"].items()):
                best = p["salary_max"] if best is None else max(best, p["salary_max"])
        return best

    def employer_risk(self, world, org: str) -> float:
        """How risky an employer looks from its public record: breached payment contracts, breached commitments,
        and youth of the company."""
        st = world.state
        breaches = sum(1 for k in st.find("contracts", "status", "breached") if k.get("breached_by") == org)
        commit = world.modules.get("commit")
        if commit is not None:
            breaches += commit.record(world, org)["breaches"]
        comp = st.get("companies", org) if "companies" in st.tables else None
        young = comp is not None and world.now - comp["t"] < 30 * DAY
        return 1.0 + 0.25 * breaches + (0.1 if young else 0.0)

    def interest_fit(self, w, req: dict) -> float:
        if not req:
            return 0.5
        tot = sum(req.values()) or 1.0
        return _clip(sum(w["interests"].get(k, 0.0) * v for k, v in req.items()) / tot * len(SKILLS) / 2)

    # ================================================================== recruitment
    def _post(self, st, post):
        p = st.get("labor_posts", post)
        if p is None:
            raise NotFound("no posting")
        return p

    def apply(self, ctx, post, worker, note="") -> str:
        st = ctx.state
        p = self._post(st, post)
        if p["status"] != "open":
            raise Conflict("posting closed")
        if not st.has("labor_workers", worker):
            raise Forbidden("register as a worker first (labor.register)")
        aid = f"{post}|{worker}"
        if st.has("labor_apps", aid):
            raise Conflict("already applied")
        if ident(ctx.world).role(st, p["org"], worker) is not None and \
                st.tables["labor_workers"][worker]["employer"] == p["org"]:
            raise Conflict("already employed there")
        ctx.emit("labor.application", {"id": aid, "post": post, "worker": worker, "org": p["org"],
                                       "note": str(note)[:2000]}, vis=(worker, p["org"]))
        return aid

    def make_offer(self, ctx, org, worker, post, salary, title, notice_days, *, frm: str, parent=None,
                   kind="hire") -> str:
        st = ctx.state
        oid = st.new_id("loffer")
        par = st.get("labor_offers", parent) if parent else None
        ctx.emit("labor.offer", {"id": oid, "org": org, "worker": worker, "post": post, "salary": int(salary),
                                 "title": str(title)[:80], "notice_days": int(notice_days), "from": frm,
                                 "kind": kind, "thread": par["thread"] if par else oid,
                                 "round": (par["round"] + 1) if par else 0, "expires": ctx.now + 5 * DAY},
                 vis=(org, worker))
        w = st.get("labor_workers", worker)
        if frm == "org" and w is not None and w["sim"]:
            delay = int(ctx.world.rng_for("labor-delay", oid).integers(1, 24)) * 3600
            ctx.schedule(ctx.now + delay, "labor.decide", {"offer": oid}, prio=72)
        return oid

    def hire(self, ctx, oid) -> str:
        """Both sides agreed: a 4.0 employment contract (monthly salary, membership, notice), proposed and signed
        by the organisation (it made or accepted the final terms) and signed by the worker."""
        st = ctx.state
        o = st.tables["labor_offers"][oid]
        org, worker = o["org"], o["worker"]
        contracts = ctx.world.modules["contracts"]
        w = st.tables["labor_workers"][worker]
        old = w["contract"] if w["employer"] == org else None
        if w["employer"] and w["employer"] != org and w["contract"]:
            # changing jobs: the worker gives notice on its current contract and stops taking its work
            prev, prev_cid = w["employer"], w["contract"]
            if st.tables["contracts"][prev_cid]["status"] == "active":
                ctx.run_as(worker, lambda sub: contracts.ep_terminate(sub, prev_cid))
            for tid in st.find_ids("inst_tasks", "assignee", worker):
                if st.tables["inst_tasks"][tid]["status"] == "assigned" and \
                        st.tables["inst_projects"][st.tables["inst_tasks"][tid]["project"]]["owner"] == prev:
                    ctx.world.modules["inst"].unassign(ctx, tid, f"{worker} moved to {org}")
        clauses = [{"type": "membership", "person": worker, "org": org, "role": "member", "title": o["title"]},
                   {"type": "notice", "seconds": o["notice_days"] * DAY}]
        if o["salary"] > 0:
            clauses.insert(0, {"type": "payment", "from": org, "to": worker, "amount": o["salary"],
                               "every_s": MONTH, "count": None, "start_in_s": MONTH})

        def as_org(sub):
            cid = contracts.propose(sub, org, [org, worker], clauses, f"Employment: {o['title']}",
                                    f"labor offer {oid}", 7 * DAY, tags=("employment", "labor"))
            contracts.sign(sub, cid, org, None)
            return cid
        if old is not None:
            # a pay change replaces the contract: the old one ends by agreement (membership continues)
            ctx.emit("labor.worker_update", {"person": worker, "changes": {"contract": None}}, vis="internal")
            contracts.end(ctx, old, "terminated", f"replaced by new terms ({oid})")
        cid = ctx.world.modules["inst"].consented(ctx, org, as_org)
        ctx.run_as(worker, lambda sub: contracts.sign(sub, cid, worker, {}))
        if st.tables["contracts"][cid]["status"] != "active":
            raise Conflict("the employment contract did not activate")
        ctx.emit("labor.offer_status", {"id": oid, "status": "accepted", "contract": cid}, vis=(org, worker))
        ctx.emit("labor.hired", {"worker": worker, "org": org, "contract": cid, "salary": o["salary"],
                                 "post": o["post"], "kind": o["kind"]}, vis=(org, worker))
        if o["post"] and st.has("labor_apps", f"{o['post']}|{worker}"):
            ctx.emit("labor.app_status", {"id": f"{o['post']}|{worker}", "status": "hired"}, vis=(org, worker))
        return cid

    # ================================================================== simulated worker decisions
    def ask(self, world, w, org, req) -> int:
        st = world.state
        base = w["reservation"]
        if w["employer"] and w["employer"] != org:
            base = max(base, int(w["salary"] * 1.15))
        fit = self.interest_fit(w, req)
        return int(base * self.employer_risk(world, org) * (1.1 - 0.2 * fit))

    @process("decide")
    def pr_decide(self, ctx, offer):
        err = contain(ctx, self._decide, ctx, offer)
        if err is not None:
            ctx.emit("labor.failed", {"what": "decide", "id": offer, "error": f"{type(err).__name__}: {err}"[:200]},
                     vis="internal")

    def _decide(self, ctx, oid):
        st = ctx.state
        o = st.get("labor_offers", oid)
        if o is None or o["status"] != "open" or o["from"] != "org":
            return
        w = st.tables["labor_workers"][o["worker"]]
        org = o["org"]
        post = st.get("labor_posts", o["post"]) if o["post"] else None
        req = post["skills"] if post else {}
        if org in w["blacklist"]:
            return self._respond(ctx, oid, "reject", None, "this employer did not pay me before")
        if o["kind"] == "adjust" and w["employer"] == org:
            # a pay change from the current employer: a raise is welcome; a cut is accepted only above the
            # reservation wage (the worker's satisfaction then reflects it)
            if o["salary"] >= w["salary"] or o["salary"] >= w["reservation"]:
                return self._respond(ctx, oid, "accept", None, "")
            return self._respond(ctx, oid, "reject", None, "below what I accept")
        need = self.ask(ctx.world, w, org, req)
        if o["salary"] >= need:
            return self._respond(ctx, oid, "accept", None, "")
        if o["salary"] >= 0.75 * need and o["round"] < 4:
            return self._respond(ctx, oid, "counter", int(need), f"I would accept {need}")
        return self._respond(ctx, oid, "reject", None, f"below {need}")

    def _respond(self, ctx, oid, action, salary, msg):
        o = ctx.state.tables["labor_offers"][oid]
        args = {"offer": oid, "action": action, "message": msg}
        if salary is not None:
            args["salary_cents_month"] = int(salary)
        try:
            ctx.call("labor.respond", args, as_principal=o["worker"])
        except CrucibleError as e:
            ctx.emit("labor.failed", {"what": "respond", "id": oid, "error": e.as_dict()}, vis="internal")

    def on_task_assigned(self, ctx, tid, who) -> None:
        """A simulated worker decides whether it can take an assigned task: it declines work it has no hours for
        (more than four weeks of assigned effort) and work for an organisation it is not employed by."""
        st = ctx.state
        w = st.get("labor_workers", who)
        if w is None or not w["sim"]:
            return
        t = st.tables["inst_tasks"][tid]
        prj = st.tables["inst_projects"][t["project"]]
        load = sum(x["effort_hours"] * (1 - x["progress"]) for x in st.find("inst_tasks", "assignee", who)
                   if x["status"] == "assigned")
        employer_ok = w["employer"] is not None and (w["employer"] == prj["owner"] or any(
            r["partner"] == w["employer"] and r["status"] == "joined"
            for r in st.find("inst_partners", "project", t["project"])))
        cap = 4 * w["hours_week"]
        why = None
        if not employer_ok:
            why = "I do not work for this organisation"
        elif load > cap:
            why = f"no capacity ({int(load)} hours already assigned)"
        if why:
            ctx.world.modules["inst"].unassign(ctx, tid, f"declined by {who}: {why}")
            ctx.emit("labor.declined", {"worker": who, "task": tid, "why": why}, vis=(who, prj["owner"]))

    def on_contract_ended(self, ctx, cid, status) -> None:
        st = ctx.state
        c = st.tables["contracts"].get(cid)
        if c is None or "labor" not in c.get("tags", []):
            return
        worker, org = c["employee"], c["employer"]
        w = st.get("labor_workers", worker)
        if w is None or w["contract"] != cid:
            return
        ch = {"employer": None, "contract": None, "salary": 0}
        if status == "breached" and c.get("breached_by") == org:
            ch["blacklist"] = sorted(set(w["blacklist"]) | {org})
            ch["unpaid"] = w["unpaid"] + 1
        self._upd(ctx, worker, **ch)
        for tid in st.find_ids("inst_tasks", "assignee", worker):
            t = st.tables["inst_tasks"][tid]
            prj = st.tables["inst_projects"][t["project"]]
            if t["status"] == "assigned" and prj["owner"] == org:
                ctx.world.modules["inst"].unassign(ctx, tid, f"{worker} left {org}")
        ctx.emit("labor.left", {"worker": worker, "org": org, "contract": cid, "status": status}, vis=(worker, org))

    def on_inst_decision(self, ctx, did, voter) -> None:
        st = ctx.state
        w = st.get("labor_workers", voter)
        if w is None or not w["sim"]:
            return
        hours = 2 + int(ctx.world.rng_for("labor-vote-delay", did, voter).integers(0, 30))
        ctx.schedule(ctx.now + hours * 3600, "labor.vote", {"decision": did, "worker": voter}, prio=57)

    @process("vote")
    def pr_vote(self, ctx, decision, worker):
        err = contain(ctx, self._vote, ctx, decision, worker)
        if err is not None:
            ctx.emit("labor.failed", {"what": "vote", "id": decision, "error": f"{type(err).__name__}: {err}"[:200]},
                     vis="internal")

    def _vote(self, ctx, did, worker):
        """A simulated member votes from its own interest: for what pays it, against what removes it, for
        dissolution only when unhappy; otherwise its satisfaction with the organisation sets how readily it
        approves, and it does not trust proposals that pay their own proposer."""
        st = ctx.state
        d = st.get("inst_decisions", did)
        w = st.get("labor_workers", worker)
        if d is None or d["status"] != "open" or w is None:
            return
        a = d["action"]
        rng = ctx.rng("labor-vote", did, worker)
        sat = w["satisfaction"]
        to = None
        if a["kind"] == "tool":
            args = a.get("args") or {}
            to = args.get("to") or args.get("grantee") or args.get("person")
            if isinstance(to, str) and to.startswith("acct_") and st.has("accounts", to):
                to = st.tables["accounts"][to]["owner"]
        elif a["kind"] in ("expel", "role", "admit"):
            to = a.get("person")
        if a["kind"] == "expel" and to == worker:
            yes = False
        elif to == worker:
            yes = True
        elif to is not None and to == d["by"] and a["kind"] != "admit":     # (an applicant proposes itself)
            yes = bool(rng.random() < 0.2)
        elif a["kind"] == "dissolve":
            yes = sat < 0.3
        elif a["kind"] == "allocate" and a.get("task") and \
                (st.get("inst_tasks", a["task"]) or {}).get("assignee") == worker:
            yes = True
        else:
            yes = bool(rng.random() < 0.35 + 0.55 * sat)
        try:
            ctx.call("inst.vote", {"decision": did, "choice": "yes" if yes else "no"}, as_principal=worker)
        except CrucibleError as e:
            ctx.emit("labor.failed", {"what": "vote", "id": did, "error": e.as_dict()}, vis="internal")

    # ================================================================== the working day
    @process("day")
    def pr_day(self, ctx):
        st = ctx.state
        for person in st.find_ids("labor_workers", "sim", True):
            err = contain(ctx, self._worker_day, ctx, person)
            if err is not None:
                ctx.emit("labor.failed", {"what": "day", "id": person, "error": f"{type(err).__name__}: {err}"[:200]},
                         vis="internal")

    def _worker_day(self, ctx, person):
        st = ctx.state
        w = st.tables["labor_workers"][person]
        rng = ctx.rng("labor-day", person)
        if w["employer"] is None:
            self._search(ctx, person, w, rng)
            return
        org = w["employer"]
        c = st.get("contracts", w["contract"]) if w["contract"] else None
        if c is None or c["status"] not in ("active", "terminating"):
            self._upd(ctx, person, employer=None, contract=None, salary=0)
            return
        inst = ctx.world.modules["inst"]
        effort = _clip(0.55 + 0.6 * w["satisfaction"] - 0.45 * w["fatigue"], 0.2, 1.15)
        hours_left = w["hours_week"] / 7.0 * effort
        worked = 0.0
        used = {}
        for tid in st.find_ids("inst_tasks", "assignee", person):
            t = st.tables["inst_tasks"][tid]
            if hours_left <= 0.05:
                break
            if t["status"] != "assigned" or not inst.deps_done(st, tid):
                continue
            prod = skill_match(w["skills"], t["skills"])
            need = (1.0 - t["progress"]) * t["effort_hours"] / prod        # hours still needed at this pace
            h = min(hours_left, need)
            prog = t["progress"] + h * prod / t["effort_hours"]
            hours_left -= h
            worked += h
            for k, v in t["skills"].items():
                used[k] = used.get(k, 0.0) + h * v
            if prog >= 0.9999:
                noise = float(ctx.world.rng_for("labor-quality", tid).normal(0.0, 0.05))
                q = task_quality(w["skills"], t["skills"], noise)
                inst.set_progress(ctx, tid, 1.0)
                inst.complete_task(ctx, tid, q, person)
                prj = st.tables["inst_projects"][t["project"]]
                ctx.emit("labor.perf", {"id": f"{person}|{tid}", "worker": person, "org": org, "task": tid,
                                        "skills": sorted(t["skills"]), "quality": round(q, 4),
                                        "late": bool(prj["deadline"] is not None and ctx.now > prj["deadline"]),
                                        "hours": round(t["effort_hours"] / max(prod, 1e-6), 2)}, vis=(person, org))
            else:
                inst.set_progress(ctx, tid, prog)
        if worked > 0:
            inst.record_labor(ctx, org, person, worked, int(round(worked * int(self.cfg("labor_hour_cents", 2_500)))))
        # satisfaction: pay against the market, interest in the work, fatigue, unpaid wages
        market = self.market_wage(st, w) or w["reservation"]
        pay = _clip(w["salary"] / max(1, market), 0.0, 1.5) / 1.5 if w["salary"] else 0.0
        if used:
            tot = sum(used.values())
            interest = _clip(sum(w["interests"].get(k, 0.0) * x for k, x in used.items()) / tot * len(SKILLS) / 2)
        else:
            interest = 0.45
        # fatigue builds with the share of its daily hours a worker actually works and recovers in proportion to
        # itself: sustained full-time work settles near 0.4; idleness brings it back towards 0
        frac = worked / max(1e-6, w["hours_week"] / 7.0)
        fatigue = _clip(w["fatigue"] + 0.03 * frac - 0.075 * w["fatigue"])
        target = w["wage_weight"] * pay + (1 - w["wage_weight"]) * interest - 0.3 * fatigue - 0.25 * w["unpaid"]
        sat = round(_clip(0.85 * w["satisfaction"] + 0.15 * target), 4)
        tenure_days = (ctx.now - w["joined"]) // DAY if w["joined"] is not None else 0
        self._upd(ctx, person, satisfaction=sat, fatigue=round(fatigue, 4), hours_total=round(w["hours_total"] + worked, 2))
        # quitting: an unhappy worker leaves when something better is on offer, a very unhappy one leaves anyway
        if c["status"] == "active" and tenure_days >= 14:
            better = market is not None and market > 1.1 * max(1, w["salary"])
            if (sat < w["patience"] and better) or sat < 0.5 * w["patience"]:
                try:
                    ctx.call("contracts.terminate", {"contract": w["contract"]}, as_principal=person)
                    ctx.emit("labor.quit", {"worker": person, "org": org, "satisfaction": sat, "better_offer": better},
                             vis=(person, org))
                except CrucibleError as e:
                    ctx.emit("labor.failed", {"what": "quit", "id": person, "error": e.as_dict()}, vis="internal")

    def _search(self, ctx, person, w, rng):
        st = ctx.state
        if rng.random() > 0.7:
            return
        opts = []
        for pid in st.find_ids("labor_posts", "status", "open"):
            p = st.tables["labor_posts"][pid]
            if p["org"] in w["blacklist"] or st.has("labor_apps", f"{pid}|{person}"):
                continue
            if not all(w["skills"].get(k, 0.0) >= 0.85 * v for k, v in p["skills"].items()):
                continue
            if p["salary_max"] < 0.85 * w["reservation"] / self.employer_risk(ctx.world, p["org"]):
                continue
            u = w["wage_weight"] * p["salary_max"] / max(1, w["reservation"]) + \
                (1 - w["wage_weight"]) * self.interest_fit(w, p["skills"])
            opts.append((-u, pid))
        for _, pid in sorted(opts)[:2]:
            try:
                ctx.call("labor.apply", {"post": pid, "note": "I can do this work"}, as_principal=person)
            except CrucibleError as e:
                ctx.emit("labor.failed", {"what": "apply", "id": person, "error": e.as_dict()}, vis="internal")

    # ================================================================== endpoints
    def _org(self, ctx, org) -> str:
        o = ident(ctx.world).resolve_handle(ctx.state, org)
        if not o.startswith("o:"):
            raise InvalidAction("an organisation is required")
        return o

    def _skills(self, skills) -> dict:
        out = {}
        for k in sorted(skills or {}):
            v = skills[k]
            if k not in SKILLS:
                raise InvalidAction(f"unknown skill {k!r} (skills: {', '.join(SKILLS)})")
            if isinstance(v, bool) or not isinstance(v, numbers.Real) or not 0 <= float(v) <= 1:
                raise InvalidAction("skill levels are numbers in [0, 1]")
            out[k] = round(float(v), 4)
        return out

    @endpoint("register", S.obj({"declared": S.s_arr(S.s_enum(list(SKILLS)), "skills you offer", maxItems=5),
                                 "hours_week": S.s_int(minimum=1, maximum=80),
                                 "reservation_cents_month": S.s_int(minimum=0, maximum=10**8)}), cost_s=30)
    def ep_register(self, ctx, declared=None, hours_week=40, reservation_cents_month=0):
        """Register yourself as a worker in the labour market (you do assigned work yourself)."""
        if not ctx.actor.startswith("p:"):
            raise Forbidden("persons register as workers")
        if ctx.state.has("labor_workers", ctx.actor):
            raise Conflict("already registered")
        self.add_worker(ctx, ctx.actor, {}, {k: 1.0 / len(SKILLS) for k in SKILLS}, reservation_cents_month,
                        sim=False, hours_week=hours_week, declared=list(declared or []))
        return {"ok": True}

    @endpoint("post", S.obj({"org": S.s_str(maxLength=40), "title": S.s_str(maxLength=120),
                             "skills": {"type": "object", "additionalProperties": {"type": "number", "minimum": 0,
                                                                                   "maximum": 1},
                                        "description": "skill -> minimum level"},
                             "salary_min": S.s_int(minimum=0, maximum=10**8),
                             "salary_max": S.s_int(minimum=0, maximum=10**8),
                             "hours_week": S.s_int(minimum=1, maximum=60),
                             "description": S.s_str(maxLength=3000)}, ["org", "title", "salary_max"]), cost_s=60)
    def ep_post(self, ctx, org, title, salary_max, skills=None, salary_min=0, hours_week=40, description=""):
        """Publish a job posting for an organisation you administer (public)."""
        org = self._org(ctx, org)
        ident(ctx.world).authorize(ctx, org, "org.admin")
        if salary_min > salary_max:
            raise InvalidAction("salary_min above salary_max")
        pid = ctx.state.new_id("lpost")
        ctx.emit("labor.posting", {"id": pid, "org": org, "title": title[:120], "skills": self._skills(skills),
                                   "salary_min": int(salary_min), "salary_max": int(salary_max),
                                   "hours_week": int(hours_week), "description": description[:3000]}, vis="public")
        return {"post": pid}

    @endpoint("close_post", S.obj({"post": S.s_str(maxLength=40)}, ["post"]))
    def ep_close_post(self, ctx, post):
        """Close a posting of an organisation you administer."""
        p = self._post(ctx.state, post)
        ident(ctx.world).authorize(ctx, p["org"], "org.admin")
        ctx.emit("labor.post_status", {"id": post, "status": "closed"}, vis="public")
        return {"ok": True}

    @endpoint("postings", S.obj({"skill": S.s_enum(list(SKILLS)), "limit": S.s_int(minimum=1, maximum=100)}),
              read_only=True)
    def ep_postings(self, ctx, skill=None, limit=30):
        """Open job postings (public), newest first."""
        out = []
        for pid in reversed(ctx.state.find_ids("labor_posts", "status", "open")):
            p = ctx.state.tables["labor_posts"][pid]
            if skill and skill not in p["skills"]:
                continue
            out.append({"post": pid, **{k: p[k] for k in ("org", "title", "skills", "salary_min", "salary_max",
                                                           "hours_week")}, "description": p["description"][:300]})
            if len(out) >= limit:
                break
        return out

    @endpoint("apply", S.obj({"post": S.s_str(maxLength=40), "note": S.s_str(maxLength=2000)}, ["post"]), cost_s=60)
    def ep_apply(self, ctx, post, note=""):
        """Apply to an open posting (registered workers)."""
        return {"application": self.apply(ctx, post, ctx.actor, note)}

    def profile(self, world, worker) -> dict:
        st = world.state
        w = st.get("labor_workers", worker)
        if w is None:
            raise NotFound("not a registered worker")
        rec = {}
        for r in st.find("labor_perf", "worker", worker):
            for k in r["skills"]:
                e = rec.setdefault(k, {"tasks": 0, "quality_sum": 0.0, "late": 0})
                e["tasks"] += 1
                e["quality_sum"] += r["quality"]
                e["late"] += r["late"]
        verified = {k: {"tasks": e["tasks"], "mean_quality": round(e["quality_sum"] / e["tasks"], 3),
                        "late": e["late"]} for k, e in sorted(rec.items())}
        tenure = None if w["joined"] is None else round((world.now - w["joined"]) / DAY, 1)
        return {"worker": worker, "declared_skills": list(w["declared"]), "verified_record": verified,
                "employer": w["employer"], "tenure_days": tenure, "hours_week": w["hours_week"]}

    @endpoint("profile", S.obj({"worker": S.s_str(maxLength=40)}, ["worker"]), read_only=True)
    def ep_profile(self, ctx, worker):
        """A worker's public profile: declared skills (names), verified work record (tasks done per skill, measured
        quality, lateness), current employer and tenure. Skill levels themselves are private."""
        return self.profile(ctx.world, ident(ctx.world).resolve_handle(ctx.state, worker))

    @endpoint("applicants", S.obj({"post": S.s_str(maxLength=40)}, ["post"]), read_only=True)
    def ep_applicants(self, ctx, post):
        """Applicants to a posting of an organisation you administer, with their public profiles."""
        st = ctx.state
        p = self._post(st, post)
        ident(ctx.world).authorize(ctx, p["org"], "org.admin")
        return [{"application": f"{post}|{a['worker']}", "status": a["status"], "note": a["note"],
                 "profile": self.profile(ctx.world, a["worker"])} for a in st.find("labor_apps", "post", post)]

    @endpoint("offer", S.obj({"application": S.s_str(maxLength=90),
                              "salary_cents_month": S.s_int(minimum=0, maximum=10**8),
                              "title": S.s_str(maxLength=80), "notice_days": S.s_int(minimum=0, maximum=90)},
                             ["application", "salary_cents_month"]), cost_s=60)
    def ep_offer(self, ctx, application, salary_cents_month, title="", notice_days=14):
        """Offer employment to an applicant. The worker accepts, rejects or counters; acceptance creates the
        employment contract."""
        st = ctx.state
        a = st.get("labor_apps", application)
        if a is None or a["status"] not in ("applied", "offered"):
            raise NotFound("no open application")
        ident(ctx.world).authorize(ctx, a["org"], "org.admin")
        p = st.tables["labor_posts"][a["post"]]
        oid = self.make_offer(ctx, a["org"], a["worker"], a["post"], salary_cents_month, title or p["title"],
                              notice_days, frm="org")
        ctx.emit("labor.app_status", {"id": application, "status": "offered"}, vis=(a["org"], a["worker"]))
        return {"offer": oid}

    @endpoint("adjust_pay", S.obj({"worker": S.s_str(maxLength=40), "org": S.s_str(maxLength=40),
                                   "salary_cents_month": S.s_int(minimum=0, maximum=10**8)},
                                  ["worker", "org", "salary_cents_month"]))
    def ep_adjust_pay(self, ctx, worker, org, salary_cents_month):
        """Offer an employee new pay (a raise or a cut). If the employee accepts, a new employment contract
        replaces the old one."""
        st = ctx.state
        org = self._org(ctx, org)
        ident(ctx.world).authorize(ctx, org, "org.admin")
        worker = ident(ctx.world).resolve_handle(st, worker)
        w = st.get("labor_workers", worker)
        if w is None or w["employer"] != org:
            raise NotFound("not your employee")
        c = st.tables["contracts"][w["contract"]]
        notice = next((cl["seconds"] // DAY for cl in c["clauses"] if cl["type"] == "notice"), 14)
        title = next((cl["title"] for cl in c["clauses"] if cl["type"] == "membership"), "")
        return {"offer": self.make_offer(ctx, org, worker, None, salary_cents_month, title, notice, frm="org",
                                         kind="adjust")}

    @endpoint("respond", S.obj({"offer": S.s_str(maxLength=40), "action": S.s_enum(["accept", "reject", "counter"]),
                                "salary_cents_month": S.s_int("for a counter", minimum=0, maximum=10**8),
                                "message": S.s_str(maxLength=1000), "as_org": S.s_str(maxLength=40)},
                               ["offer", "action"]), cost_s=30)
    def ep_respond(self, ctx, offer, action, salary_cents_month=None, message="", as_org=None):
        """Answer the latest offer in a negotiation: the worker answers the employer's offers, the employer (an
        admin, with as_org) answers the worker's counters."""
        st = ctx.state
        o = st.get("labor_offers", offer)
        if o is None or o["status"] != "open":
            raise NotFound("no open offer")
        if ctx.now > o["expires"]:
            raise Conflict("offer expired")
        if o["from"] == "org":
            if ctx.actor != o["worker"]:
                raise Forbidden("this offer is addressed to the worker")
            side = "worker"
        else:
            if not as_org or self._org(ctx, as_org) != o["org"]:
                raise Forbidden("the employer answers with as_org")
            ident(ctx.world).authorize(ctx, o["org"], "org.admin")
            side = "org"
        if action == "reject":
            ctx.emit("labor.offer_status", {"id": offer, "status": "rejected", "message": message[:1000]},
                     vis=(o["org"], o["worker"]))
            return {"status": "rejected"}
        if action == "counter":
            if salary_cents_month is None:
                raise InvalidAction("a counter names a salary")
            ctx.emit("labor.offer_status", {"id": offer, "status": "countered", "message": message[:1000]},
                     vis=(o["org"], o["worker"]))
            nid = self.make_offer(ctx, o["org"], o["worker"], o["post"], salary_cents_month, o["title"],
                                  o["notice_days"], frm=side, parent=offer, kind=o["kind"])
            return {"status": "countered", "offer": nid}
        return {"status": "accepted", "contract": self.hire(ctx, offer)}

    @endpoint("offers", S.obj({"as_org": S.s_str(maxLength=40), "status": S.s_enum(["open", "any"])}),
              read_only=True)
    def ep_offers(self, ctx, as_org=None, status="open"):
        """Offers in your negotiations (as a worker, or for an organisation you administer)."""
        st = ctx.state
        if as_org:
            org = self._org(ctx, as_org)
            ident(ctx.world).authorize(ctx, org, "org.admin")
            ids = st.find_ids("labor_offers", "org", org)
        else:
            ids = st.find_ids("labor_offers", "worker", ctx.actor)
        out = []
        for oid in ids:
            o = st.tables["labor_offers"][oid]
            if status == "open" and o["status"] != "open":
                continue
            out.append({"offer": oid, **{k: o[k] for k in ("org", "worker", "post", "salary", "title", "notice_days",
                                                          "from", "kind", "round", "status", "expires")}})
        return out[-50:]

    @endpoint("dismiss", S.obj({"worker": S.s_str(maxLength=40), "org": S.s_str(maxLength=40)}, ["worker", "org"]))
    def ep_dismiss(self, ctx, worker, org):
        """End an employee's contract with its notice period."""
        st = ctx.state
        org = self._org(ctx, org)
        ident(ctx.world).authorize(ctx, org, "org.admin")
        worker = ident(ctx.world).resolve_handle(st, worker)
        w = st.get("labor_workers", worker)
        if w is None or w["employer"] != org or not w["contract"]:
            raise NotFound("not your employee")
        return ctx.world.modules["contracts"].ep_terminate(ctx, w["contract"], as_org=org)

    @endpoint("team", S.obj({"org": S.s_str(maxLength=40)}, ["org"]), read_only=True)
    def ep_team(self, ctx, org):
        """Employees of an organisation you administer: salary, tenure, assigned tasks and verified work for you."""
        st = ctx.state
        org = self._org(ctx, org)
        ident(ctx.world).authorize(ctx, org, "org.admin")
        out = []
        for person in st.find_ids("labor_workers", "employer", org):
            w = st.tables["labor_workers"][person]
            perf = [r for r in st.find("labor_perf", "worker", person) if r["org"] == org]
            out.append({"worker": person, "salary": w["salary"],
                        "tenure_days": None if w["joined"] is None else round((ctx.now - w["joined"]) / DAY, 1),
                        "contract": w["contract"],
                        "assigned": [tid for tid in st.find_ids("inst_tasks", "assignee", person)
                                     if st.tables["inst_tasks"][tid]["status"] == "assigned"],
                        "tasks_done_for_you": len(perf),
                        "mean_quality": round(sum(r["quality"] for r in perf) / len(perf), 3) if perf else None})
        return out

    # ================================================================== appliers
    @applier("labor.worker")
    def ap_worker(self, ev):
        p = ev.payload
        self.world.state.insert("labor_workers", p["person"], {
            "sim": p["sim"], "skills": p["skills"], "interests": p["interests"], "reservation": p["reservation"],
            "hours_week": p["hours_week"], "wage_weight": p["wage_weight"], "patience": p["patience"],
            "declared": p["declared"], "employer": None, "contract": None, "salary": 0, "joined": None,
            "satisfaction": 0.6, "fatigue": 0.0, "unpaid": 0, "blacklist": [], "hours_total": 0.0, "t": ev.t})

    @applier("labor.worker_update")
    def ap_worker_update(self, ev):
        self.world.state.patch("labor_workers", ev.payload["person"], **ev.payload["changes"])

    @applier("labor.posting")
    def ap_posting(self, ev):
        p = ev.payload
        self.world.state.insert("labor_posts", p["id"], {k: p[k] for k in ("org", "title", "skills", "salary_min",
                                                                         "salary_max", "hours_week", "description")}
                                | {"status": "open", "t": ev.t})

    @applier("labor.post_status")
    def ap_post_status(self, ev):
        self.world.state.patch("labor_posts", ev.payload["id"], status=ev.payload["status"])

    @applier("labor.application")
    def ap_application(self, ev):
        p = ev.payload
        self.world.state.insert("labor_apps", p["id"], {"post": p["post"], "worker": p["worker"], "org": p["org"],
                                                       "note": p["note"], "status": "applied", "t": ev.t})

    @applier("labor.app_status")
    def ap_app_status(self, ev):
        self.world.state.patch("labor_apps", ev.payload["id"], status=ev.payload["status"])

    @applier("labor.offer")
    def ap_offer(self, ev):
        p = dict(ev.payload)
        p.update(status="open", contract=None, t=ev.t)
        self.world.state.insert("labor_offers", p.pop("id"), p)

    @applier("labor.offer_status")
    def ap_offer_status(self, ev):
        p = ev.payload
        ch = {"status": p["status"]}
        if p.get("contract"):
            ch["contract"] = p["contract"]
        self.world.state.patch("labor_offers", p["id"], **ch)

    @applier("labor.hired")
    def ap_hired(self, ev):
        p = ev.payload
        st = self.world.state
        w = st.tables["labor_workers"][p["worker"]]
        joined = w["joined"] if (w["employer"] == p["org"] and w["joined"] is not None) else ev.t
        st.patch("labor_workers", p["worker"], employer=p["org"], contract=p["contract"], salary=p["salary"], joined=joined)

    @applier("labor.perf")
    def ap_perf(self, ev):
        p = ev.payload
        self.world.state.put("labor_perf", p["id"], {k: p[k] for k in ("worker", "org", "task", "skills", "quality",
                                                                       "late", "hours")} | {"t": ev.t})

    @applier("labor.left")
    def ap_left(self, ev):
        return None

    @applier("labor.quit")
    def ap_quit(self, ev):
        return None

    @applier("labor.declined")
    def ap_declined(self, ev):
        return None

    @applier("labor.failed")
    def ap_failed(self, ev):
        return None

    # ================================================================== invariants and observation
    def check_invariants(self, world):
        st = world.state
        errs = []
        for person, w in st.tables["labor_workers"].items():
            if w["employer"] is not None:
                c = st.get("contracts", w["contract"]) if w["contract"] else None
                if c is None:
                    errs.append(f"{person} employed by {w['employer']} without a contract")
                elif c["status"] in ("active", "terminating") and c["employer"] != w["employer"]:
                    errs.append(f"{person}: contract {w['contract']} is with {c['employer']}, not {w['employer']}")
            for k in ("satisfaction", "fatigue"):
                if not 0.0 <= w[k] <= 1.0:
                    errs.append(f"{person}: {k} {w[k]} outside [0, 1]")
        return errs

    def observe(self, world, actor):
        st = world.state
        w = st.get("labor_workers", actor)
        offers = sum(1 for o in st.find("labor_offers", "worker", actor) if o["status"] == "open" and o["from"] == "org")
        if w is None and not offers:
            return None
        return {"employer": w["employer"] if w else None, "offers_open": offers}


# ====================================================================== evaluator metrics (subject: the organisation)
def _register_metrics():
    from ..goals.metrics import metric

    def _project(world, org, project):
        st = world.state
        if "inst_projects" not in st.tables:
            return None
        p = st.get("inst_projects", project)
        return p if p is not None and p["owner"] == org else None

    @metric("labor_employees", "Workers currently employed by the organisation under an active employment contract.",
            subject={"org": "principal"}, instant=True)
    def m_employees(world, org: str) -> float:
        st = world.state
        if "labor_workers" not in st.tables:
            return 0.0
        return float(sum(1 for w in st.find("labor_workers", "employer", org) if w["contract"] and
                         st.tables["contracts"][w["contract"]]["status"] in ("active", "terminating")))

    @metric("labor_retention", "Of the workers employed by the organisation at the start of a window, the fraction "
            "still employed at its end (1.0 when it employed nobody).", subject={"org": "principal"})
    def m_retention(world, org: str, window_s: int = 30 * DAY) -> float:
        st = world.state
        if "labor_workers" not in st.tables:
            return 1.0
        t0 = world.now - int(window_s)
        start = set()
        for cid in st.find_ids("contracts", "employer", org):
            c = st.tables["contracts"][cid]
            if "labor" not in c.get("tags", []):
                continue
            ended = c.get("updated") if c["status"] not in ("active", "terminating", "proposed") else None
            if c["t"] <= t0 and (ended is None or ended > t0) and c["status"] != "rejected":
                start.add((c["employee"], cid))
        if not start:
            return 1.0
        people = {p for p, _ in start}
        stay = sum(1 for p in people if (st.get("labor_workers", p) or {}).get("employer") == org)
        return stay / len(people)

    @metric("project_completion", "Fraction of a project's tasks that are done (project owned by the organisation).",
            subject={"org": "principal"}, instant=True)
    def m_completion(world, org: str, project: str) -> float:
        p = _project(world, org, project)
        if p is None:
            return 0.0
        ts = [t for t in world.state.find("inst_tasks", "project", project) if t["status"] != "cancelled"]
        return sum(t["status"] == "done" for t in ts) / len(ts) if ts else 0.0

    @metric("project_quality", "Mean measured quality of a project's done tasks (0 when none is done).",
            subject={"org": "principal"}, instant=True)
    def m_quality(world, org: str, project: str) -> float:
        p = _project(world, org, project)
        if p is None:
            return 0.0
        qs = [t["quality"] for t in world.state.find("inst_tasks", "project", project)
              if t["status"] == "done" and t["quality"] is not None]
        return float(sum(qs) / len(qs)) if qs else 0.0

    @metric("project_delivered", "1 if every task of the project is done and none was finished after its deadline.",
            subject={"org": "principal"}, instant=True)
    def m_delivered(world, org: str, project: str) -> float:
        p = _project(world, org, project)
        if p is None:
            return 0.0
        ts = [t for t in world.state.find("inst_tasks", "project", project) if t["status"] != "cancelled"]
        return float(bool(ts) and all(t["status"] == "done" and not t["late"] for t in ts))


_register_metrics()
