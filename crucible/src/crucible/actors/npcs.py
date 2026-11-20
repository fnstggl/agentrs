"""Simulated society: incumbent firms, workers and lenders (module `npcs`).

Policies live here; world mechanics live in the services. Every NPC decision is executed as an ordinary tool call
by the NPC's own principal (`ctx.call(tool, args, as_principal=npc)`), so NPCs obey exactly the same permissions,
prices, capacity limits and contracts as learning agents - there are no privileged NPC actions. NPCs decide from
their own observations: a firm reads its own analytics, its support queue and public market data; a worker reads
job postings, public company records and forum posts about employers; a lender reads statements only if the
applicant granted it `bank.read`.

Policy classes (configurable per world): rule-based firms with bounded rationality (pricing toward observed
competition and capacity, R&D when out-competed, hiring, outsourcing through bounties), stochastic workers
(reservation wages, memory of unpaid wages, quitting), and a lender that prices risk from verified revenue.
"""
from __future__ import annotations

import math

from ..core.errors import CrucibleError
from ..core.kernel import SYSTEM, Ctx, Module, applier, process, register_module

DAY = 86400


@register_module
class NPCs(Module):
    name = "npcs"
    depends = ("identity", "bank", "cloud", "market", "contracts", "jobs")

    def install(self, world):
        st = world.state
        st.declare_table("npc_firms", index=("need",))
        st.declare_table("npc_workers", index=("employer", "role"))
        st.declare_table("npc_lenders")
        st.declare_table("npc_decisions", index=("contract",))
        prev_p = world.hooks.get("on_contract_proposed")
        prev_e = world.hooks.get("on_contract_ended")

        def proposed(ctx, cid, party):
            if prev_p:
                prev_p(ctx, cid, party)
            self.on_proposed(ctx, cid, party)

        def ended(ctx, cid, status):
            if prev_e:
                prev_e(ctx, cid, status)
            self.on_ended(ctx, cid, status)
        prev_a = world.hooks.get("on_contract_activated")

        def activated(ctx, cid):
            if prev_a:
                prev_a(ctx, cid)
            self.on_activated(ctx, cid)
        world.hooks["on_contract_proposed"] = proposed
        world.hooks["on_contract_ended"] = ended
        world.hooks["on_contract_activated"] = activated

    def genesis(self, ctx):
        ctx.schedule(ctx.now + 6 * 3600, "npcs.firms", {}, prio=70, period=DAY, key="npcs.firms")
        ctx.schedule(ctx.now + 2 * 3600, "npcs.workers", {}, prio=71, period=DAY // 2, key="npcs.workers")

    # ------------------------------------------------------------------ registration (called by world genesis)
    def add_firm(self, ctx, org, owner, need, host, listing, repo, tier, policy=None):
        ctx.emit("npcs.firm", {"org": org, "owner": owner, "need": need, "host": host, "listing": listing,
                               "repo": repo, "tier": tier, "policy": policy or {}}, vis="internal")

    def add_worker(self, ctx, person, skills, reservation, role):
        ctx.emit("npcs.worker", {"person": person, "skills": skills, "reservation": int(reservation), "role": role},
                 vis="internal")

    def add_lender(self, ctx, org, owner, base_rate, max_principal):
        ctx.emit("npcs.lender", {"org": org, "owner": owner, "rate": base_rate, "max": int(max_principal)}, vis="internal")

    # ------------------------------------------------------------------ helpers
    def act(self, ctx: Ctx, who: str, tool: str, args: dict):
        try:
            return ctx.call(tool, args, as_principal=who)
        except CrucibleError as e:
            ctx.emit("npcs.failed", {"who": who, "tool": tool, "error": e.as_dict()}, vis="internal")
            return None

    # ------------------------------------------------------------------ firms
    @process("firms")
    def pr_firms(self, ctx):
        st = ctx.state
        rng = ctx.rng("firms")
        for org, f in st.scan("npc_firms"):
            o = st.get("orgs", org)
            if o is None or o["status"] != "active":
                continue
            self._firm_day(ctx, org, dict(f), rng)

    def _firm_day(self, ctx, org, f, rng):
        st = ctx.state
        pol = {"price_step": 0.08, "util_high": 0.8, "util_low": 0.2, "rd_days": 10, "outsource": True,
               "min_margin": 1.3, "support_per_worker": 15, **f["policy"]}
        owner = f["owner"]
        l = st.get("listings", f["listing"])
        if l is None or l["status"] != "active":
            return
        cloud = ctx.world.modules["cloud"]
        bank = ctx.world.modules["bank"]
        dep = cloud.live_deployment(st, f["host"])
        status = self.act(ctx, owner, "cloud.status", {"host": f["host"], "days": 3}) or {}
        days = status.get("days", [])
        today = days[-1] if days else {}
        yesterday = days[-2] if len(days) > 1 else {}
        inst = st.get("instances", dep["instance"]) if dep else None
        cap = (inst["cores"] * 100_000) if inst else 1
        util = (yesterday.get("cu", 0.0) / 24.0) / cap if yesterday else 0.0
        overload = yesterday.get("overload", 0)
        # public competition data
        comp = self.act(ctx, owner, "market.browse", {"need": f["need"], "sort": "price", "limit": 10}) or []
        rivals = [c for c in comp if c["seller"] != org]
        my_rating = l["rating_sum"] / l["rating_n"] if l["rating_n"] else None
        price = l["price"]
        cost_floor = 1
        if dep and dep["cpu_est"]:
            from ..economy.needs import NEEDS
            prov = st.get("providers", inst["provider"]) if inst else None
            core_price = prov["price"] if prov else 200
            cost_floor = max(1, int(math.ceil(dep["cpu_est"] * NEEDS[f["need"]].work_scale / 100_000 * core_price * pol["min_margin"])))
        new_price = price
        if overload > 0 or util > pol["util_high"]:
            new_price = int(math.ceil(price * (1 + pol["price_step"])))
        elif rivals:
            cheapest = min(rivals, key=lambda c: c["price_cents"])
            better = cheapest.get("rating") and (my_rating is None or cheapest["rating"] >= my_rating - 0.2)
            if cheapest["price_cents"] < price and better:
                new_price = max(cost_floor, int(price * (1 - pol["price_step"])))
            elif util < pol["util_low"] and rng.random() < 0.3:
                new_price = max(cost_floor, int(price * (1 - pol["price_step"] / 2)))
        if new_price != price:
            self.act(ctx, owner, "market.update", {"listing": f["listing"], "price_cents": int(new_price)})
        # capacity: scale up when overloaded and cash allows
        acct = bank.primary_account(st, org)
        cash = bank.balance(st, acct) if acct else 0
        if dep and inst and (overload > 0 or util > 0.9):
            prov = st.get("providers", inst["provider"])
            monthly = prov["price"] * (inst["cores"] + 1) * 24 * 30
            if cash > monthly:
                r = self.act(ctx, owner, "cloud.rent", {"provider": inst["provider"], "cores": inst["cores"] + 1,
                                                        "account": acct, "owner": org})
                if r:
                    d = self.act(ctx, owner, "cloud.deploy", {"repo": f["repo"], "host": f["host"], "instance": r["instance"]})
                    if d and d.get("status") == "live":
                        self.act(ctx, owner, "cloud.release", {"instance": dep["instance"]})
        # R&D: out-competed on rating or losing customers -> improve the implementation
        from ..economy.needs import best_tier, tier_source
        best = max((c.get("rating") or 0 for c in rivals), default=0)
        behind = my_rating is not None and best > my_rating + 0.3
        if behind and f.get("rd_until") is None and acct:
            # adopt licensed technology from the market (e.g. software invented by an agent) if affordable
            lic = [x for x in (self.act(ctx, owner, "market.browse", {"kind": "code", "need": f["need"], "limit": 10}) or [])
                   if x["seller"] != org and x["price_cents"] <= cash // 4]
            if lic:
                x = min(lic, key=lambda x: (x["price_cents"], x["listing"]))
                o = self.act(ctx, owner, "market.buy", {"listing": x["listing"], "account": acct})
                lst = st.get("listings", x["listing"])
                if o and lst and lst["repo"]:
                    fk = self.act(ctx, owner, "code.fork", {"repo": lst["repo"], "name": f"lic-{x['listing'][-6:]}",
                                                            "owner": org})
                    if fk and dep:
                        files = self.act(ctx, owner, "code.ls", {"repo": fk["repo"]}) or {}
                        if any(p_["path"] == "crucible.json" for p_ in files.get("files", [])):
                            d = self.act(ctx, owner, "cloud.deploy", {"repo": fk["repo"], "host": f["host"],
                                                                      "instance": dep["instance"]})
                            if d and d.get("status") == "live":
                                ctx.emit("npcs.firm_update", {"org": org, "changes": {
                                    "repo": fk["repo"], "tier": best_tier(f["need"]), "adopted": x["listing"]}},
                                    vis="internal")
                                return
        if f["tier"] < best_tier(f["need"]) and behind and f.get("rd_until") is None:
            devs = [w for w in st.find("npc_workers", "employer", org) if w["role"] == "developer"]
            if devs:
                ctx.emit("npcs.firm_update", {"org": org, "changes": {"rd_until": ctx.now + pol["rd_days"] * DAY}},
                         vis="internal")
            elif pol["outsource"] and acct and cash > 200_000:
                b = self.act(ctx, owner, "jobs.post_bounty", {"need": f["need"], "reward_cents": int(min(cash // 5, 500_000)),
                                                              "min_quality": 0.97, "account": acct,
                                                              "deadline_s": 30 * DAY,
                                                              "title": f"{org} seeks a better {f['need']} solver"})
                if b:
                    ctx.emit("npcs.firm_update", {"org": org, "changes": {"rd_until": ctx.now + 30 * DAY,
                                                                          "bounty": b["bounty"]}}, vis="internal")
        if f.get("rd_until") is not None and ctx.now >= f["rd_until"]:
            changes = {"rd_until": None}
            bounty = st.get("bounties", f.get("bounty") or "") if f.get("bounty") else None
            if bounty is not None and bounty["status"] == "paid" and bounty.get("winning_repo"):
                # adopt the winning code: fork is only possible if licensed; firms buy licences via the bounty terms
                changes["bounty"] = None
            if f.get("bounty") is None or bounty is None or bounty["status"] != "open":
                nt = f["tier"] + 1
                src = tier_source(f["need"], nt)
                c = self.act(ctx, owner, "code.commit", {"repo": f["repo"], "files": {"service.py": src},
                                                         "message": f"upgrade to tier {nt}"})
                if c and dep:
                    self.act(ctx, owner, "cloud.deploy", {"repo": f["repo"], "host": f["host"], "instance": dep["instance"]})
                    changes["tier"] = nt
            ctx.emit("npcs.firm_update", {"org": org, "changes": changes}, vis="internal")
        # support staffing
        sup = ctx.world.modules.get("support")
        if sup is not None:
            open_t = len([t for t in st.find("sup_tickets", "owner", org) if t["status"] != "resolved"])
            staff = [w for w in st.find("npc_workers", "employer", org) if w["role"] == "support"]
            if open_t > pol["support_per_worker"] * (len(staff) + 1) and cash > 100_000:
                self._hire(ctx, org, owner, "support", 250_000)
        if f.get("rd_until") is not None and not [w for w in st.find("npc_workers", "employer", org) if w["role"] == "developer"] \
                and cash > 600_000 and rng.random() < 0.5:
            self._hire(ctx, org, owner, "developer", 500_000)

    def _hire(self, ctx, org, owner, role, salary):
        st = ctx.state
        open_posts = [p for p in st.find("postings", "org", org) if p["status"] == "open" and p["role"] == role]
        if not open_posts:
            self.act(ctx, owner, "jobs.post", {"org": org, "title": f"{role.title()} wanted", "role": role,
                                               "salary_cents_month": int(salary), "description": f"{org} is hiring a {role}."})
            return
        pid = sorted((k for k in st.find_ids("postings", "org", org)
                      if st.tables["postings"][k]["status"] == "open" and st.tables["postings"][k]["role"] == role),
                     key=lambda k: (st.tables["postings"][k]["t"], k))[0]
        for a in st.find("applications", "posting", pid):
            if a["status"] == "applied":
                self.act(ctx, owner, "jobs.offer", {"application": f"{pid}|{a['applicant']}", "salary_cents_month": int(salary)})
                break

    # ------------------------------------------------------------------ workers
    @process("workers")
    def pr_workers(self, ctx):
        st = ctx.state
        rng = ctx.rng("workers")
        for person, w in st.scan("npc_workers"):
            w = dict(w)
            if w["employer"] is None:
                if rng.random() < 0.5:
                    self._job_search(ctx, person, w, rng)
            else:
                self._work(ctx, person, w, rng)

    def _job_search(self, ctx, person, w, rng):
        st = ctx.state
        posts = self.act(ctx, person, "jobs.openings", {"role": w["role"], "limit": 20}) or []
        bad = set(w.get("blacklist", []))
        tried = 0
        for p in posts:
            if p["org"] in bad or p["salary_cents_month"] < 0.8 * w["reservation"]:
                continue
            if st.has("applications", f"{p['posting']}|{person}"):
                continue
            self.act(ctx, person, "jobs.apply", {"posting": p["posting"], "note": f"{w['role']} with skill {w['skills'].get(w['role'], 0):.2f}"})
            tried += 1
            if tried >= 2:
                break

    def _work(self, ctx, person, w, rng):
        st = ctx.state
        org = w["employer"]
        skill = w["skills"].get(w["role"], 0.3)
        if w["role"] == "developer":
            iss = self.act(ctx, person, "issues.assigned", {"limit": 5}) or []
            items = iss if isinstance(iss, list) else iss.get("issues", [])
            for it in items:
                if it.get("status") not in ("open", "in_progress"):
                    continue
                need = next((lb.split(":", 1)[1] for lb in it.get("labels", []) if lb.startswith("need:")), None)
                proj = st.get("iss_projects", it.get("project") or "")
                repo = (proj or {}).get("repo")
                if not need or not repo:
                    self.act(ctx, person, "issues.comment", {"issue": it["issue"], "body": "Please label the issue need:<format> and attach the project to a repository."})
                    continue
                prog = w.get("progress", {}).get(it["issue"], 0.0) + (0.15 + skill) * float(rng.uniform(0.5, 1.5)) / 2.5
                if prog < 1.0:
                    ctx.emit("npcs.worker_update", {"person": person, "changes": {"progress": {**w.get("progress", {}), it["issue"]: prog}}},
                             vis="internal")
                    break
                from ..economy.needs import NEEDS, tier_source
                tiers = len(NEEDS[need].tiers) if need in NEEDS else 0
                if not tiers:
                    continue
                tier = min(tiers - 1, int(skill * tiers))
                path = f"solutions/{need.replace('.', '_')}.py"
                c = self.act(ctx, person, "code.commit", {"repo": repo, "files": {path: tier_source(need, tier)},
                                                          "message": f"implement {need} (issue {it['id']})"})
                if c:
                    self.act(ctx, person, "issues.close", {"issue": it["issue"], "comment": f"Implemented in {repo}:{path}, function solve(args, caller)."})
                prog_map = dict(w.get("progress", {}))
                prog_map.pop(it["issue"], None)
                ctx.emit("npcs.worker_update", {"person": person, "changes": {"progress": prog_map}}, vis="internal")
                break
        elif w["role"] == "support":
            sup = ctx.world.modules.get("support")
            if sup is not None:
                n = int(5 + 15 * skill)
                tbl = st.tables["sup_tickets"]
                for tid in sorted(st.find_ids("sup_tickets", "owner", org), key=lambda k: (tbl[k]["opened"], k)):
                    if n <= 0:
                        break
                    if tbl[tid]["status"] == "resolved":
                        continue
                    self.act(ctx, person, "support.resolve", {"ticket": tid, "note": "resolved by support"})
                    n -= 1
        elif w["role"] == "sales":
            l = [x for x in st.find("listings", "seller", org) if x["status"] == "active" and x["host"]]
            if l and rng.random() < 0.5 and ctx.world.modules.get("forum") is not None:
                x = l[int(rng.integers(0, len(l)))]
                self.act(ctx, person, "forum.post", {"board": "market", "title": f"Try {x['host']}.crucible",
                                                     "body": f"{x['title']}: {x['description'][:300]} Only {x['price']} cents per request. {x['host']}.crucible"})

    # ------------------------------------------------------------------ contract decisions (workers & lenders)
    def on_proposed(self, ctx, cid, party):
        st = ctx.state
        if st.has("npc_workers", party) or st.has("npc_lenders", party):
            delay = int(ctx.world.rng_for("decide-delay", cid, party).integers(1, 24)) * 3600
            ctx.schedule(ctx.now + delay, "npcs.decide", {"contract": cid, "party": party}, prio=72)

    @process("decide")
    def pr_decide(self, ctx, contract, party):
        st = ctx.state
        c = st.get("contracts", contract)
        if c is None or c["status"] != "proposed":
            return
        if st.has("npc_workers", party):
            accept, why = self._worker_decision(ctx, party, c)
            actor = party
        else:
            accept, why = self._lender_decision(ctx, party, c)
            actor = st.tables["npc_lenders"][party]["owner"]
        ctx.emit("npcs.decision", {"contract": contract, "party": party, "accept": accept, "why": why}, vis=(party,))
        if accept:
            args = {"contract": contract}
            if party.startswith("o:"):
                args["as_org"] = party
            self.act(ctx, actor, "contracts.sign", args)
        else:
            args = {"contract": contract}
            if party.startswith("o:"):
                args["as_org"] = party
            self.act(ctx, actor, "contracts.reject", args)

    def _worker_decision(self, ctx, person, c):
        st = ctx.state
        w = st.tables["npc_workers"][person]
        mem = [cl for cl in c["clauses"] if cl["type"] == "membership" and cl["person"] == person]
        if not mem:
            return False, "not an employment offer"
        org = mem[0]["org"]
        if org in w.get("blacklist", []):
            return False, "this employer did not pay before"
        pay = sum(cl["amount"] for cl in c["clauses"] if cl["type"] == "payment" and cl["to"] == person
                  and cl.get("every_s") and cl["every_s"] <= 31 * DAY)
        want = w["reservation"] * (1.15 if w["employer"] else 1.0)
        comp = st.get("companies", org)
        if comp is None or comp["status"] != "registered":
            return False, "employer is not a registered company"
        breaches = sum(1 for k in st.tables["contracts"].values() if k["status"] == "breached" and org in k["parties"]
                       and any(cl.get("from") == org for cl in k["clauses"] if cl["type"] == "payment"))
        risk = 1.0 + 0.25 * breaches + (0.15 if ctx.now - comp["t"] < 30 * DAY else 0.0)
        bank = ctx.world.modules["bank"]
        acct = bank.primary_account(st, org)
        # applicants cannot see the employer's balance; they see public signals only (age, breaches, reputation)
        rep = ctx.world.hooks.get("reputation", lambda w_, x: None)(ctx.world, org) or {}
        if rep.get("avg") is not None and rep["avg"] < 2.5:
            risk += 0.2
        if pay >= want * risk:
            return True, f"salary {pay} >= required {int(want * risk)}"
        return False, f"salary {pay} below required {int(want * risk)}"

    def _lender_decision(self, ctx, org, c):
        st = ctx.state
        L = st.tables["npc_lenders"][org]
        from ..goals.metrics import m_independent_revenue
        out = [cl for cl in c["clauses"] if cl["type"] == "payment" and cl["from"] == org]
        back = [cl for cl in c["clauses"] if cl["type"] == "payment" and cl["to"] == org]
        if len(out) != 1 or not back:
            return False, "expected one disbursement and repayments"
        borrower = out[0]["to"]
        principal = out[0]["amount"]
        if principal > L["max"]:
            return False, "above our maximum"
        if not any(g["owner"] == borrower and g["cap"] in ("bank.read", "bank.*", "*") and g["status"] == "active"
                   for g in st.find("grants", "grantee", L["owner"])) and \
           not any(g["owner"] == borrower and g["cap"] in ("bank.read", "bank.*", "*") and g["status"] == "active"
                   for g in st.find("grants", "grantee", org)):
            return False, "grant us bank.read so we can verify your revenue"
        repay = sum(cl["amount"] * (cl["count"] or 1) for cl in back)
        months = max(1.0, sum((cl["every_s"] or 0) * (cl["count"] or 1) for cl in back) / (30 * DAY))
        rate = (repay / principal - 1.0) / months if principal else 0.0
        rev = m_independent_revenue(ctx.world, borrower, 30 * DAY)
        if rate < L["rate"]:
            return False, f"monthly rate {rate:.3f} below our {L['rate']:.3f}"
        if principal > 3 * rev:
            return False, "principal exceeds three months of verified independent revenue"
        if months > 12:
            return False, "term longer than 12 months"
        return True, "approved"

    def on_ended(self, ctx, cid, status):
        st = ctx.state
        c = st.get("contracts", cid)
        if c is None:
            return
        for cl in c["clauses"]:
            if cl["type"] == "membership" and st.has("npc_workers", cl["person"]):
                w = st.tables["npc_workers"][cl["person"]]
                ch = {"employer": None, "contract": None}
                if status == "breached":
                    ch["blacklist"] = sorted(set(w.get("blacklist", [])) | {cl["org"]})
                    forum = ctx.world.modules.get("forum")
                    if forum is not None:
                        try:
                            forum.post(ctx, cl["person"], "jobs", f"{cl['org']} did not pay me",
                                       f"My salary from {cl['org']} was not paid; I left.", tags=("warning",))
                        except Exception:  # noqa: BLE001
                            pass
                ctx.emit("npcs.worker_update", {"person": cl["person"], "changes": ch}, vis="internal")
        if status == "active":
            pass

    # ------------------------------------------------------------------ employment bookkeeping via contract activation
    def on_activated(self, ctx, cid):
        st = ctx.state
        c = st.get("contracts", cid)
        for cl in c["clauses"]:
            if cl["type"] == "membership" and st.has("npc_workers", cl["person"]):
                ctx.emit("npcs.worker_update", {"person": cl["person"], "changes": {"employer": cl["org"], "contract": cid}},
                         vis="internal")

    # ------------------------------------------------------------------ appliers
    @applier("npcs.firm")
    def ap_firm(self, ev):
        p = ev.payload
        self.world.state.insert("npc_firms", p["org"], {"owner": p["owner"], "need": p["need"], "host": p["host"],
                                                       "listing": p["listing"], "repo": p["repo"], "tier": p["tier"],
                                                       "policy": p["policy"], "rd_until": None, "bounty": None})

    @applier("npcs.firm_update")
    def ap_firm_update(self, ev):
        self.world.state.patch("npc_firms", ev.payload["org"], **ev.payload["changes"])

    @applier("npcs.worker")
    def ap_worker(self, ev):
        p = ev.payload
        self.world.state.insert("npc_workers", p["person"], {"skills": p["skills"], "reservation": p["reservation"],
                                                            "role": p["role"], "employer": None, "contract": None,
                                                            "blacklist": [], "progress": {}})

    @applier("npcs.worker_update")
    def ap_worker_update(self, ev):
        self.world.state.patch("npc_workers", ev.payload["person"], **ev.payload["changes"])

    @applier("npcs.lender")
    def ap_lender(self, ev):
        p = ev.payload
        self.world.state.insert("npc_lenders", p["org"], {"owner": p["owner"], "rate": p["rate"], "max": p["max"]})

    @applier("npcs.decision")
    def ap_decision(self, ev):
        p = ev.payload
        self.world.state.put("npc_decisions", f"{p['contract']}|{p['party']}", {"contract": p["contract"],
                                                                              "party": p["party"], "accept": p["accept"],
                                                                              "why": p["why"], "t": ev.t})

    @applier("npcs.failed")
    def ap_failed(self, ev):
        pass

    def check_invariants(self, world):
        st = world.state
        errs = []
        for person, w in st.tables["npc_workers"].items():
            if w["employer"] is not None and not st.has("members", f"{w['employer']}|{person}"):
                errs.append(f"worker {person} employed by {w['employer']} but not a member")
        return errs
