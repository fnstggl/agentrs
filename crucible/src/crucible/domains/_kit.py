"""Shared parts of the labour-marketplace and organisational-ecosystem domains (workstream I3).

* `spawn_worker`: a simulated worker of the 4.0/5.0 labour market (`labor`) with private skills drawn from a
  configurable skill mix (which skills are common or scarce is part of a domain configuration's structure);
* `make_brief`: a project brief - a task graph of a given topology (pipeline, wide, diamond, chain) with skill
  requirements, effort and dependencies - and its critical path;
* `ToolPolicy`: the base of the evaluator-side scripted reference policies. A reference acts only through ordinary
  tools as its participant: it reads with read-only tools (`World.query`, the participant's own permissions,
  nothing recorded) and acts with one recorded tool call per turn. It never reads world state directly.
* `TeamOps`: the scripted reference's team-building and coordination steps (recruit by declared skills and
  verified record, assign ready tasks, keep pay at the public market, subcontract to partner organisations).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..agents.base import Policy
from ..commerce.labor import SKILLS, _clip
from ..core.errors import CrucibleError
from .firms import declared_level, est_fit

DAY = 86400


# ====================================================================== workers
def spawn_worker(ctx, handle: str, rng, *, weights: dict | None = None, caps: dict | None = None,
                 lo: int = 300_000, hi: int = 900_000, savings: int = 150_000) -> str:
    """A simulated worker (person of kind npc, own account and savings) with two strong skills drawn by `weights`
    (skill -> relative frequency; default uniform), the rest weak; `caps` bounds a skill's level (a skill the open
    market does not supply). Interests favour the strong skills; the reservation wage rises with skill."""
    from ..services.identity import ident
    labor = ctx.world.modules["labor"]
    bank = ctx.world.modules["bank"]
    pid = ident(ctx.world).create_person(ctx, handle, handle, "npc", {"role": "worker"})
    acct = bank.open_account(ctx, pid, "checking")
    if savings:
        bank.move(ctx, "acct_genesis", acct, int(savings), "endowment", f"savings of {pid}")
    wts = np.array([float((weights or {}).get(k, 1.0)) for k in SKILLS])
    wts = wts / wts.sum()
    strong = set(int(j) for j in rng.choice(len(SKILLS), size=2, replace=False, p=wts))
    skills = {}
    for j, k in enumerate(SKILLS):
        lvl = float(rng.uniform(0.55, 0.95)) if j in strong else float(rng.uniform(0.05, 0.45))
        if caps and k in caps:
            lvl = min(lvl, float(caps[k]))
        skills[k] = round(lvl, 3)
    raw = rng.dirichlet(np.ones(len(SKILLS)) * 0.7) + np.array([0.15 if j in strong else 0.0
                                                                for j in range(len(SKILLS))])
    raw = raw / raw.sum()
    interests = {k: round(float(raw[j]), 3) for j, k in enumerate(SKILLS)}
    top = sorted(SKILLS, key=lambda k: -skills[k])
    level = float(np.mean([skills[k] for k in top[:2]]))
    reservation = int(lo + (hi - lo) * _clip((level - 0.55) / 0.4) * float(rng.uniform(0.85, 1.15)))
    labor.add_worker(ctx, pid, skills, interests, reservation, sim=True, hours_week=int(rng.integers(32, 46)),
                     wage_weight=round(float(rng.uniform(0.4, 0.85)), 3),
                     patience=round(float(rng.uniform(0.3, 0.45)), 3), declared=top[:3])
    return pid


# ====================================================================== project briefs
def _req(rng, skills, levels, k):
    pick = sorted(rng.choice(sorted(skills), size=min(k, len(skills)), replace=False))
    return {s: round(float(rng.uniform(*levels)), 3) for s in pick}


def make_brief(rng, topology: str, skills: list[str], *, levels=(0.35, 0.6), effort=(24, 70), stages: int = 3,
               width: int = 6, special: dict | None = None) -> dict:
    """A task graph. Each task: key, title, skills {skill: level}, effort hours, deps (keys of earlier tasks) and
    stage (milestones group tasks by stage; the stage's last task is its deliverable). `special` adds the
    requirement {skill: level} to one task in the middle of the graph (a capability few can supply)."""
    tasks: list[dict] = []

    def add(title, deps, stage, k=1):
        key = f"t{len(tasks)}"
        tasks.append({"key": key, "title": title, "skills": _req(rng, skills, levels, k),
                      "effort": int(rng.integers(effort[0], effort[1] + 1)), "deps": list(deps), "stage": stage})
        return key

    if topology == "pipeline":          # stages of two parallel tracks joined by an integration task
        prev = None
        for s in range(stages):
            a = add(f"stage {s + 1} track A", [prev] if prev else [], s)
            b = add(f"stage {s + 1} track B", [prev] if prev else [], s)
            prev = add(f"stage {s + 1} integration", [a, b], s, k=2)
    elif topology == "chain":           # a strict sequence: every task depends on the previous one
        prev = None
        for s in range(stages):
            for j in range(2):
                prev = add(f"step {s + 1}.{j + 1}", [prev] if prev else [], s, k=1 + (j == 1))
    elif topology == "wide":            # kickoff, many independent work packages, integration, launch
        k0 = add("kickoff", [], 0)
        mids = [add(f"work package {j + 1}", [k0], 0) for j in range(width)]
        integ = add("integration", mids, 1, k=2)
        add("launch", [integ], 2)
    elif topology == "diamond":         # two branches with a cross-dependency, joined at the end
        k0 = add("kickoff", [], 0)
        a1 = add("branch A design", [k0], 0)
        b1 = add("branch B design", [k0], 0)
        a2 = add("branch A build", [a1], 1)
        b2 = add("branch B build", [b1, a1], 1, k=2)
        a3 = add("branch A hardening", [a2], 1)
        j = add("integration", [a3, b2], 2, k=2)
        add("release", [j], 2)
    else:
        raise ValueError(f"unknown topology {topology!r}")
    if special:
        mid = tasks[len(tasks) // 2]
        if mid["key"] == tasks[-1]["key"]:
            mid = tasks[-2]
        mid["skills"] = dict(special)
        mid["title"] = mid["title"] + " (specialist)"
    by = {t["key"]: t for t in tasks}
    finish: dict[str, float] = {}
    for t in tasks:
        finish[t["key"]] = t["effort"] + max([finish[d] for d in t["deps"]] or [0.0])
    for t in tasks:
        t["cp_hours"] = finish[t["key"]]
    sinks = {}
    for t in tasks:
        sinks[t["stage"]] = t["key"]
    return {"tasks": tasks, "cp_hours": max(finish.values()), "total_hours": sum(t["effort"] for t in tasks),
            "stage_sinks": [sinks[s] for s in sorted(sinks)], "by_key": by}


# ====================================================================== tool-only reference policies
class ToolPolicy(Policy):
    """Scripted reference: reads through read-only tools as its participant, acts one recorded tool call a turn.

    `plan()` returns the list of actions to take next (it may read with `q`); they are executed in order, one per
    turn, and the policy plans again when the queue is empty and `every` seconds have passed."""
    kind = "scripted"
    every = 3 * 3600

    def __init__(self, world, principal: str):
        self.world = world
        self.me = principal
        self.queue: list[dict] = []
        self.next_plan = 0
        self.failures: list[dict] = []
        self.log: list[str] = []

    def q(self, tool: str, **args) -> Any:
        try:
            return self.world.query(self.me, tool, args)
        except CrucibleError as e:
            return {"error": e.as_dict()}

    def now(self) -> int:
        return self.world.now

    def act(self, obs, tools):
        if not self.queue and self.world.now >= self.next_plan:
            self.queue = list(self.plan() or [])
            self.next_plan = self.world.now + self.every
        if self.queue:
            return self.queue.pop(0)
        return None

    def observe_result(self, action, result):
        if not result.get("ok"):
            self.failures.append({"t": self.world.now, "action": action, "error": result.get("error")})
            return
        tag = action.get("tag")
        if tag:
            for h in list(self.handlers):
                h(tag, result.get("result"))

    @property
    def handlers(self) -> list:
        if not hasattr(self, "_handlers"):
            self._handlers = []
        return self._handlers

    def plan(self) -> list[dict]:
        raise NotImplementedError


def A(tool: str, tag: str | None = None, **args) -> dict:
    """An action; `tag` (ignored by the world) names what the policy does with its result."""
    out = {"tool": tool, "args": args}
    if tag:
        out["tag"] = tag
    return out


class TeamOps:
    """Team-building and coordination steps of the scripted references (they use public information only:
    postings, declared skills, verified records, project state, offers, organisation profiles)."""

    def __init__(self, pol: ToolPolicy, org: str, project: str):
        self.p, self.org, self.project = pol, org, project
        self.posts: dict[str, dict] = {}          # skill -> {post, t, salary, open}
        self.offered: set[str] = set()
        self.last_raise = -10 * DAY
        self.contracted: dict[str, str] = {}      # task -> partner org it is contracted to
        self.sub_offers: dict[str, dict] = {}     # offer -> {org, tasks, amount, open}
        self._pending: dict[str, dict] = {}       # tag -> what the tagged action was for
        pol.handlers.append(self.on_result)

    def on_result(self, tag: str, result) -> None:
        rec = self._pending.pop(tag, None)
        if rec is None or not isinstance(result, dict):
            return
        if rec["kind"] == "post":
            self.posts[rec["skill"]] = {"post": result["post"], "t": self.p.now(), "salary": rec["salary"],
                                        "open": True}
        elif rec["kind"] == "offer":
            self.sub_offers[result["offer"]] = {"org": rec["org"], "tasks": rec["tasks"], "amount": rec["amount"],
                                                "open": True}

    def _tag(self, kind: str, **rec) -> str:
        tag = f"{kind}:{len(self._pending)}:{self.p.now()}:{sorted(rec.items())!r}"[:200]
        self._pending[tag] = dict(rec, kind=kind)
        return tag

    # ------------------------------------------------------------------ reading
    def project_view(self) -> dict:
        return self.p.q("inst.project", project=self.project)

    def team(self) -> list[dict]:
        r = self.p.q("labor.team", org=self.org)
        return r if isinstance(r, list) else []

    def declared(self, worker: str) -> list:
        r = self.p.q("labor.profile", worker=worker)
        return list(r.get("declared_skills") or []) if isinstance(r, dict) else []

    @staticmethod
    def ready(tasks: dict, t: dict, lookahead: float = 0.0) -> bool:
        return all(tasks[d]["status"] == "done" or (lookahead and tasks[d]["status"] == "assigned" and
                                                     tasks[d]["progress"] >= lookahead)
                   for d in t["depends_on"])

    def market_salary(self, skill: str) -> int:
        rows = self.p.q("labor.postings", skill=skill, limit=100)
        best = 0
        for r in rows if isinstance(rows, list) else []:
            if r["org"] != self.org:
                best = max(best, int(r["salary_max"]))
        return best

    # ------------------------------------------------------------------ recruitment
    def staffing_need(self, tasks: dict, days_left: float, covered: set[str]) -> dict[str, int]:
        """Workers wanted per main skill: remaining effort of open tasks needing it over what one worker does in
        the time left (4.5 effective hours a day), with a margin."""
        need: dict[str, float] = {}
        for tid, t in tasks.items():
            if t["status"] in ("done", "cancelled") or tid in self.contracted:
                continue
            main = max(sorted(t["skills"]), key=lambda k: t["skills"][k]) if t["skills"] else None
            if main is None or main not in covered:
                continue
            need[main] = need.get(main, 0.0) + t["effort_hours"] * (1 - t["progress"])
        out = {}
        for k, h in sorted(need.items()):
            out[k] = max(1, int(math.ceil(h / max(1.0, 4.5 * max(3.0, days_left - 3)) * 1.35)))
        return out

    def recruit(self, tasks: dict, days_left: float, salary_cap: int, covered: set[str]) -> list[dict]:
        acts: list[dict] = []
        team = self.team()
        strengths: dict[str, int] = {}
        for m in team:
            for k in self.declared(m["worker"])[:2]:
                strengths[k] = strengths.get(k, 0) + 1
        need = self.staffing_need(tasks, days_left, covered)
        for skill, n in sorted(need.items()):
            have = strengths.get(skill, 0)
            post = self.posts.get(skill)
            if have >= n:
                if post and post.get("open"):
                    acts.append(A("labor.close_post", post=post["post"]))
                    post["open"] = False
                continue
            if post is None or not post.get("open"):
                sal = int(min(salary_cap, max(650_000, 1.1 * self.market_salary(skill))))
                acts.append(A("labor.post", tag=self._tag("post", skill=skill, salary=sal), org=self.org,
                              title=f"{skill} specialist", skills={skill: 0.5}, salary_min=300_000, salary_max=sal,
                              hours_week=40))
                continue
            if self.p.now() - post["t"] < 2 * DAY:
                continue
            apps = self.p.q("labor.applicants", post=post["post"])
            cands = []
            for a in apps if isinstance(apps, list) else []:
                prof = a["profile"]
                if a["status"] != "applied" or prof["employer"] is not None or a["application"] in self.offered:
                    continue
                dec = prof["declared_skills"]
                if skill not in dec[:2]:
                    continue
                rec = prof["verified_record"].get(skill, {}).get("mean_quality")
                cands.append((-(declared_level(dec, skill) + (rec or 0.0)), a["application"]))
            for _, app in sorted(cands)[:max(0, n - have)]:
                self.offered.add(app)
                acts.append(A("labor.offer", application=app, salary_cents_month=int(post["salary"] * 0.85),
                              title=f"{skill} specialist"))
        offers = self.p.q("labor.offers", as_org=self.org)
        for o in offers if isinstance(offers, list) else []:
            if o["from"] == "worker" and o["status"] == "open":
                ok = o["salary"] <= salary_cap
                acts.append(A("labor.respond", offer=o["offer"], action="accept" if ok else "reject", as_org=self.org))
        return acts

    # ------------------------------------------------------------------ assignment
    def assign(self, tasks: dict, people: dict[str, list], loads: dict[str, float], min_fit: float = 0.85,
               lookahead: float = 0.6) -> list[dict]:
        """Assign open tasks whose dependencies are done (or nearly) to the best-fitting person with the least
        queued work (`people`: person -> declared skills)."""
        acts = []
        order = sorted(tasks, key=lambda tid: (-tasks[tid].get("cp", 0), tid))
        for tid in order:
            t = tasks[tid]
            if t["status"] != "open" or not self.ready(tasks, t, lookahead):
                continue
            best = None
            for who, dec in sorted(people.items()):
                fit = est_fit(dec, t["skills"])
                if fit < min_fit or loads.get(who, 0.0) > 60:
                    continue
                key = (-fit, loads.get(who, 0.0), who)
                if best is None or key < best[0]:
                    best = (key, who)
            if best is not None:
                who = best[1]
                acts.append(A("inst.task_assign", task=tid, assignee=who))
                loads[who] = loads.get(who, 0.0) + t["effort_hours"]
        return acts

    def keep_pay(self, team: list[dict], cash: int, payroll: int) -> list[dict]:
        """Weekly: match the best public posting for an employee's declared strengths (retention)."""
        if self.p.now() - self.last_raise < 7 * DAY:
            return []
        self.last_raise = self.p.now()
        acts = []
        room = cash - 2 * payroll
        for m in team:
            dec = self.declared(m["worker"])
            market = max([self.market_salary(k) for k in dec[:2]] or [0])
            if market > 1.05 * m["salary"] and room > 3 * (market - m["salary"]):
                acts.append(A("labor.adjust_pay", worker=m["worker"], org=self.org,
                              salary_cents_month=int(market * 1.02)))
                room -= 3 * (market - m["salary"])
        return acts

    # ------------------------------------------------------------------ subcontracting to partner organisations
    def partner_staff(self, partner: str) -> dict[str, list]:
        prof = self.p.q("id.profile", principal=partner)
        out = {}
        for m in prof.get("members", []) if isinstance(prof, dict) else []:
            if m["role"] == "member":
                dec = self.declared(m["person"])
                if dec:
                    out[m["person"]] = dec
        return out

    def subcontract(self, tasks: dict, partners: list[str], own_skills: set[str], deadline: int,
                    price_per_hour: int, budget: int, as_org: str | None = None) -> list[dict]:
        """Offer a commitment to the partner organisation whose staff fits best for each open task the team
        cannot do well, one milestone per task, due by the deadline."""
        acts = []
        open_offer_tasks = {t for o in self.sub_offers.values() if o.get("open") for t in o["tasks"]}
        by_org: dict[str, list[str]] = {}
        for tid, t in sorted(tasks.items()):
            if t["status"] != "open" or tid in self.contracted or tid in open_offer_tasks:
                continue
            if any(est_fit([k], {k: t["skills"][k]}) >= 0.85 for k in own_skills) and \
                    all(k in own_skills for k in t["skills"]):
                continue
            best = None
            for org in partners:
                staff = self.partner_staff(org)
                fit = max([est_fit(dec, t["skills"]) for dec in staff.values()] or [0.0])
                if fit >= 0.85 and (best is None or fit > best[0]):
                    best = (fit, org)
            if best is not None:
                by_org.setdefault(best[1], []).append(tid)
        for org, tids in sorted(by_org.items()):
            ms = []
            for tid in tids[:8]:
                t = tasks[tid]
                amount = int(t["effort_hours"] * price_per_hour)
                ms.append({"amount_cents": amount, "deliverable": {"kind": "task", "task": tid, "min_quality": 0.0},
                           "due_in_s": int(max(3 * DAY, deadline - self.p.now()))})
            total = sum(m["amount_cents"] for m in ms)
            if total > budget:
                continue
            terms = {"title": f"work on {self.project}", "payer": "$me", "milestones": ms}
            args = {"to": org, "terms": terms, "message": "we would like your team to take these tasks"}
            if as_org:
                args["as_org"] = as_org
            tids = [m["deliverable"]["task"] for m in ms]
            acts.append(A("commit.offer", tag=self._tag("offer", org=org, tasks=tids, amount=total), **args))
            for t in tids:                                   # not offered twice while this offer is out
                self.contracted.setdefault(t, f"pending:{org}")
        return acts

    def follow_offers(self, budget: int, as_org: str | None = None) -> list[dict]:
        """Accept a counter-offer within budget; note accepted ones."""
        acts = []
        args = {"as_org": as_org} if as_org else {}
        inbox = self.p.q("commit.inbox", status="any", **args)
        mine = {o["offer"]: o for o in inbox} if isinstance(inbox, list) else {}
        for oid, rec in list(self.sub_offers.items()):
            o = mine.get(oid)
            if o is None or not rec.get("open"):
                continue
            if o["status"] == "accepted":
                rec["open"] = False
                for t in rec["tasks"]:
                    self.contracted[t] = rec["org"]
            elif o["status"] in ("rejected", "expired", "withdrawn"):
                rec["open"] = False
            elif o["status"] == "countered":
                rec["open"] = False
                for x in mine.values():
                    if x["direction"] == "received" and x["status"] == "open" and x["thread"] == o["thread"]:
                        total = sum(m["amount_cents"] for m in x["terms"]["milestones"])
                        if total <= budget:
                            acts.append(A("commit.respond", offer=x["offer"], action="accept", **args))
                            for t in rec["tasks"]:
                                self.contracted[t] = rec["org"]
                        else:
                            acts.append(A("commit.respond", offer=x["offer"], action="reject", **args))
        return acts
