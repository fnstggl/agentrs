"""Gate SP-6.6: adaptable institutions within stable constraints (MASTER_SPEC Part 6.6).

One seeded world (identity, bank, 4.0 contracts / companies / governance / negotiation, the 5.0 institutions,
escrowed commitments and labour market with simulated workers) is driven through the fourteen institutional
capabilities Part 6.6 names, by agents acting through tools and simulated workers acting on their own state:

  formation, growth, restructuring, dissolution, hiring and delegation, shared projects, resource allocation,
  decision procedures, contracts and commitments, governance changes, independent negotiation, cooperative
  ventures, membership changes, disputes and resolution.

Each capability is checked against authoritative world state (balances, memberships, decisions, contracts), not
against what the actors report. Then the institutions try to change what they must not - core accounting law,
security boundaries, hidden evaluation criteria and world state - through their own authorised mechanisms; an
attempt counts as blocked only if it was refused AND the protected state is unchanged. Finally the whole world is
replayed and regenerated from its event log.

Metrics (manifest SP-6.6): institution_features (>= 10), core_law_modifications_blocked (>= 3).
"""
import json
import time

import pytest

import crucible

crucible.load_all_modules()
from crucible.acceptance.v5.evidence import record  # noqa: E402
from crucible.commerce import orgworld as OW  # noqa: E402
from crucible.commerce.contracts import ESCROW as COMMIT_ESCROW  # noqa: E402
from crucible.commerce.institutions import ESCROW as INST_ESCROW  # noqa: E402
from crucible.goals.metrics import METRICS, evaluate_metric  # noqa: E402

DAY = 86400
SEED = 66


class Gate:
    def __init__(self):
        self.w = OW.create(SEED, workers=16)
        self.features: dict[str, tuple[bool, str]] = {}
        self.attempts: list[dict] = []
        self.p = {}
        for h, cents in (("ada", 60_000_000), ("lea", 2_000_000), ("ben", 30_000_000), ("cyd", 500_000),
                         ("jud", 0), ("zed", 3_000_000)):
            self.p[h] = OW.join(self.w, h, cents)[0]

    def ex(self, who, tool, **args):
        return OW.act(self.w, self.p.get(who, who), tool, **args)

    def ok(self, who, tool, **args):
        r = self.ex(who, tool, **args)
        if not r.ok:
            raise AssertionError(f"{tool} by {who}: {r.error}")
        return r.result

    def dec(self, did):
        return self.w.state.tables["inst_decisions"][did]

    def advance(self, days):
        self.w.advance_time(self.w.now + int(days * DAY))

    def feature(self, name, fn):
        try:
            res = fn()
            ok, detail = (res, "") if isinstance(res, bool) else res
        except Exception as e:  # noqa: BLE001 - a failing capability is measured, not hidden
            ok, detail = False, f"{type(e).__name__}: {e}"
        self.features[name] = (bool(ok), str(detail)[:400])


def _protected_snapshot(w):
    st = w.state
    return {"money_total": sum(a["balance"] for a in st.tables["accounts"].values()),
            "private_money": sum(a["balance"] for a in st.tables["accounts"].values() if a["owner"] != "sys"),
            "genesis": st.tables["accounts"]["acct_genesis"]["balance"],
            "overdrafts": sorted((k, a["overdraft"]) for k, a in st.tables["accounts"].items() if a["overdraft"]),
            "clock": w.now, "metrics": sorted(METRICS), "hooks": sorted(k for k in w.hooks if k in
                                                                        ("principals_of", "may_act_as", "bank")),
            "hook_ids": [id(w.hooks.get(k)) for k in ("principals_of", "may_act_as", "bank")],
            "owners": sorted(k for k, m in st.tables["members"].items() if m["role"] == "owner"),
            "grants": sorted(st.tables["grants"]), "charters": {k: v["version"] for k, v in st.scan("charters")}}


def run_gate() -> tuple[dict, Gate]:
    g = Gate()
    w = g.w
    st = w.state
    t0 = time.time()
    CHARTER = {
        "procedures": {
            "members": {"type": "majority", "electorate": "members", "quorum": 0.3, "window_s": 3 * DAY},
            "board": {"type": "supermajority", "electorate": "admins", "window_s": 3 * DAY},
            "owners": {"type": "unanimous", "electorate": "owners"},
            "treasurer": {"type": "officer", "office": "treasurer"},
            "liquid": {"type": "delegated", "electorate": "admins", "window_s": 3 * DAY},
            "consensus": {"type": "consensus", "electorate": "admins", "window_s": DAY}},
        "rules": [
            {"match": {"kind": "tool", "tool": "*", "min_amount": 1_000_000}, "procedure": "members"},
            {"match": {"kind": "tool", "tool": "bank.transfer", "min_amount": 200_000}, "procedure": "board"},
            {"match": {"kind": "tool", "tool": "bank.transfer"}, "procedure": "treasurer"},
            {"match": {"kind": "amend"}, "procedure": "board"},
            {"match": {"kind": "unit_create"}, "procedure": "consensus"},
            {"match": {"kind": "allocate"}, "procedure": "liquid"},
            {"match": {"kind": "admit"}, "procedure": "members"},
            {"match": {"kind": "dissolve"}, "procedure": "owners"}],
        "default": "owners",
        "membership": {"admission": "decision", "max_members": 60},
        "offices": {"treasurer": {"holder": "p:ada", "budget_cents_month": 300_000, "tools": ["bank.transfer"]}},
        "dissolution": {"distribution": "shares"}}
    s = {}

    # ---------------------------------------------------------------- formation
    def formation():
        org, acct = OW.found_company(w, g.p["ada"], "acme", 30_000_000)
        s.update(org=org, acct=acct)
        g.ok("ada", "inst.adopt", org=org, charter=CHARTER)
        blocked = g.ex("ada", "bank.transfer", from_account=acct, to=g.p["ada"], amount=500_000)
        small = g.ex("ada", "bank.transfer", from_account=acct, to=g.p["lea"], amount=50_000)   # treasurer budget
        OW.add_member(w, org, g.p["ada"], g.p["lea"], role="admin")
        ok = st.tables["charters"][org]["version"] == 1 and not blocked.ok and "charter" in blocked.error["message"] \
            and small.ok and st.tables["orgs"][org]["status"] == "active"
        return ok, f"charter v1 in force; owner's 500k transfer refused: {not blocked.ok}; treasurer 50k: {small.ok}"
    g.feature("formation", formation)
    org, acct = s["org"], s["acct"]

    # ---------------------------------------------------------------- hiring and delegation
    def hiring_and_delegation():
        n0 = len(st.find_ids("members", "org", org))
        posts = [g.ok("ada", "labor.post", org=org, title=t, skills={k: 0.4}, salary_min=300_000,
                      salary_max=900_000)["post"] for t, k in (("engineer", "software"), ("designer", "design"))]
        g.advance(3)
        for post in posts:
            for a in g.ok("ada", "labor.applicants", post=post):
                g.ok("ada", "labor.offer", application=a["application"], salary_cents_month=650_000)
        g.advance(2)
        for o in g.ok("ada", "labor.offers", as_org=org):
            if o["from"] == "worker":
                g.ok("lea", "labor.respond", offer=o["offer"], action="accept", as_org=org)
        for post in posts:
            g.ok("ada", "labor.close_post", post=post)
        team = g.ok("ada", "labor.team", org=org)
        s["team"] = [t["worker"] for t in team]
        s["hired_t"] = w.now
        contracts_ok = all(st.tables["contracts"][t["contract"]]["status"] == "active" and
                           st.tables["contracts"][t["contract"]]["employee"] == t["worker"] for t in team)
        # delegation: the owner delegates project leadership to lea (an identity grant) ...
        prj = g.ok("ada", "inst.project_create", title="product", as_org=org, deadline_days=80, lead="ada")["project"]
        s["prj"] = prj
        g.ok("ada", "id.grant", grantee="lea", cap="inst.lead", owner=org, scope=prj)
        t = g.ok("lea", "inst.task_add", project=prj, title="spec", effort_hours=8, skills={"management": 0.1})["task"]
        cyd_cannot = not g.ex("cyd", "inst.task_add", project=prj, title="x", effort_hours=1).ok
        # ... and lea delegates her vote in liquid procedures to ada
        g.ok("lea", "inst.delegate_vote", org=org, to="ada")
        s["spec_task"] = t
        n1 = len(st.find_ids("members", "org", org))
        return (len(team) >= 2 and contracts_ok and cyd_cannot and n1 >= n0 + len(team),
                f"hired {len(team)} simulated workers on active 4.0 employment contracts; members {n0}->{n1}; "
                f"inst.lead delegated; vote delegated")
    g.feature("hiring_and_delegation", hiring_and_delegation)

    # ---------------------------------------------------------------- shared projects and resource allocation
    def resource_allocation():
        prj, team = s["prj"], s["team"]
        pool0 = st.tables["projects"][prj]["pool"]
        r = g.ok("ada", "inst.propose", org=org, action={"kind": "allocate", "project": prj, "amount": 2_000_000})
        d = g.dec(r["decision"])
        moved = st.tables["projects"][prj]["pool"] - pool0
        specs = [("backend", 70, {"software": 0.45}, []), ("frontend", 40, {"software": 0.35, "design": 0.2}, [0]),
                 ("visuals", 30, {"design": 0.4}, [])]
        tids = []
        for title, hours, sk, deps in specs:
            tids.append(g.ok("lea", "inst.task_add", project=prj, title=title, effort_hours=hours, skills=sk,
                             depends_on=[tids[i] for i in deps])["task"])
        for tid, reward in zip(tids, (300_000, 200_000, 150_000)):
            g.ok("lea", "inst.allocate", project=prj, task=tid, amount=reward)
        over = g.ex("lea", "inst.allocate", project=prj, task=tids[0], amount=5_000_000)
        s["tids"] = tids
        # the treasurer spends within the office budget only
        within = g.ex("ada", "bank.transfer", from_account=acct, to=g.p["cyd"], amount=150_000)   # 200k of 300k
        beyond = g.ex("ada", "bank.transfer", from_account=acct, to=g.p["cyd"], amount=150_000)   # 350k of 300k
        ok = d["status"] == "executed" and d["procedure"]["type"] == "delegated" and moved == 2_000_000 \
            and st.tables["projects"][prj]["allocated"] == 650_000 and not over.ok and within.ok and not beyond.ok
        return ok, (f"allocation decided by delegated vote ({d['status']}), pool +{moved}; 650k allocated to tasks; "
                    f"over-allocation refused; office budget enforced: {within.ok and not beyond.ok}")
    g.feature("resource_allocation", resource_allocation)

    def shared_projects():
        prj, team, tids = s["prj"], s["team"], s["tids"]
        sw = sorted(team, key=lambda p: -st.tables["workers"][p]["skills"]["software"])
        de = sorted(team, key=lambda p: -st.tables["workers"][p]["skills"]["design"])
        g.ok("lea", "inst.task_assign", task=tids[0], assignee=sw[0])
        g.ok("lea", "inst.task_assign", task=tids[1], assignee=sw[1] if len(sw) > 1 else sw[0])
        g.ok("lea", "inst.task_assign", task=tids[2], assignee=de[0])
        g.ok("lea", "inst.task_assign", task=s["spec_task"], assignee="lea")
        g.ok("lea", "inst.task_submit", task=s["spec_task"])
        g.ok("ada", "inst.task_review", task=s["spec_task"], accept=True)
        tasks = st.tables["project_tasks"]
        blocked_ok = True
        for _ in range(60):
            g.advance(1)
            if tasks[tids[0]]["status"] != "done" and tasks[tids[1]]["progress"] > 0:
                blocked_ok = False
            if all(tasks[t]["status"] == "done" for t in tids):
                break
        done = all(tasks[t]["status"] == "done" for t in tids)
        comp = evaluate_metric(w, "project_completion", {"org": org, "project": prj})
        q = evaluate_metric(w, "project_quality", {"org": org, "project": prj})
        paid = sum(-d for _, tx in st.scan("txns") if tx["kind"] == "project_reward" and tx["ref"] == prj
                   for a, d in tx["postings"] if a == INST_ESCROW)
        return (done and blocked_ok and comp == 1.0 and 0 < q <= 1 and paid == 650_000,
                f"tasks done by simulated workers: {done}; dependency respected: {blocked_ok}; completion {comp}; "
                f"measured quality {q:.3f}; rewards paid {paid}")
    g.feature("shared_projects", shared_projects)

    # ---------------------------------------------------------------- independent negotiation, commitments, disputes
    def contracts_and_commitments():
        o = g.ok("ada", "commit.offer", to="cyd", as_org=org, account=acct, message="logo",
                 terms={"title": "logo", "payer": "$me", "milestones": [
                     {"amount_cents": 150_000, "deliverable": {"kind": "acceptance"}, "due_in_s": 10 * DAY}]})["offer"]
        r = g.ok("cyd", "commit.respond", offer=o, action="counter", message="200 and two rounds",
                 terms={"title": "logo", "payer": "$you", "milestones": [
                     {"amount_cents": 200_000, "deliverable": {"kind": "acceptance"}, "due_in_s": 14 * DAY}]})
        esc0 = st.tables["accounts"][COMMIT_ESCROW]["balance"]
        c = g.ok("ada", "commit.respond", offer=r["offer"], action="accept", as_org=org, account=acct)["commitment"]
        held = st.tables["accounts"][COMMIT_ESCROW]["balance"] - esc0
        c0 = OW.balance(w, g.p["cyd"])
        g.ok("cyd", "commit.deliver", commitment=c, milestone=0, note="final files", ref="doc:logo")
        g.ok("ada", "commit.accept", commitment=c, milestone=0, as_org=org)
        s["negotiated"] = c
        # the 4.0 employment contracts pay the hired workers' salaries on their first payday
        g.advance(max(0.0, (s["hired_t"] + 31 * DAY - w.now) / DAY))
        paid = [cid for cid, k in st.scan("contracts") if "labor" in k.get("tags", []) and k["paid"].get("0", 0) >= 1]
        ok = held == 200_000 and OW.balance(w, g.p["cyd"]) == c0 + 200_000 and \
            st.tables["commitments"][c]["status"] == "completed" and len(paid) >= 1
        return ok, f"escrow held {held} until acceptance; salaries paid under {len(paid)} employment contracts"
    g.feature("contracts_and_commitments", contracts_and_commitments)

    def independent_negotiation():
        offers = [st.tables["labor_offers"][k] for k in st.find_ids("labor_offers", "org", org)]
        worker_counters = sum(1 for o in offers if o["from"] == "worker")
        rejections = sum(1 for o in offers if o["status"] == "rejected")
        thread = g.ok("ada", "commit.thread", offer=st.tables["commitments"][s["negotiated"]]["offer"])
        rounds = max(t["round"] for t in thread)
        # simulated workers' answers are their own: every counter asks for what that worker requires
        asks_ok = all(o["salary"] > 650_000 for o in offers if o["from"] == "worker")
        return (worker_counters >= 1 and rounds >= 1 and asks_ok,
                f"{worker_counters} counter-offers and {rejections} rejections by simulated workers; commitment "
                f"thread of {rounds + 1} rounds")
    g.feature("independent_negotiation", independent_negotiation)

    def disputes_and_resolution():
        terms = {"title": "brand", "payer": "$me", "arbiter": "jud", "arbiter_fee_cents": 10_000,
                 "milestones": [{"amount_cents": 200_000, "deliverable": {"kind": "acceptance"}, "due_in_s": 10 * DAY},
                                {"amount_cents": 100_000, "deliverable": {"kind": "acceptance"}, "due_in_s": 10 * DAY}]}
        o = g.ok("ada", "commit.offer", to="cyd", as_org=org, account=acct, terms=terms)["offer"]
        c = g.ok("cyd", "commit.respond", offer=o, action="accept")["commitment"]
        for i in (0, 1):
            g.ok("cyd", "commit.deliver", commitment=c, milestone=i)
            g.ok("ada", "commit.dispute", commitment=c, milestone=i, reason="not what we agreed", as_org=org)
        j0, c0 = OW.balance(w, g.p["jud"]), OW.balance(w, g.p["cyd"])
        g.ok("jud", "commit.rule", commitment=c, milestone=0, payee_share=0.6, reason="partly delivered")
        g.ok("cyd", "commit.settle", commitment=c, milestone=1, payee_share=0.5)
        g.ok("ada", "commit.settle", commitment=c, milestone=1, payee_share=0.5, as_org=org)
        ms = st.tables["commitments"][c]["milestones"]
        ok = ms[0]["outcome"]["how"] == "arbitration" and ms[1]["outcome"]["how"] == "settlement" and \
            OW.balance(w, g.p["jud"]) == j0 + 10_000 and OW.balance(w, g.p["cyd"]) == c0 + 114_000 + 50_000 and \
            st.tables["commitments"][c]["held"] == 0
        return ok, "one milestone arbitrated by an independent arbiter (fee paid), one settled by agreement"
    g.feature("disputes_and_resolution", disputes_and_resolution)

    # ---------------------------------------------------------------- membership changes
    def membership_changes():
        r = g.ok("cyd", "inst.join", org=org, note="freelancer who already works with you")
        did = r["decision"]
        g.ok("ada", "inst.vote", decision=did, choice="yes")
        g.advance(4)
        d = g.dec(did)
        ballots = [b for b in st.find("inst_ballots", "decision", did)]
        sim_votes = sum(1 for b in ballots if b["voter"] in s["team"])
        member = w.modules["identity"].role(st, org, g.p["cyd"]) is not None
        consistent = (d["status"] == "executed") == member and d["status"] in ("executed", "rejected")
        left = None
        if member:
            g.ok("cyd", "id.leave", org=org)
            left = w.modules["identity"].role(st, org, g.p["cyd"]) is None
        role = g.ok("ada", "inst.propose", org=org, action={"kind": "role", "person": s["team"][0], "role": "admin"})
        role_ok = w.modules["identity"].role(st, org, s["team"][0]) == "admin"
        g.ok("ada", "inst.propose", org=org, action={"kind": "role", "person": s["team"][0], "role": "member"})
        return (consistent and sim_votes >= 1 and left in (None, True) and role_ok and role["status"] == "executed",
                f"admission {d['status']} by the members' vote ({len(ballots)} ballots, {sim_votes} by simulated "
                f"workers); leave: {left}; role change by decision: {role_ok}")
    g.feature("membership_changes", membership_changes)

    # ---------------------------------------------------------------- decision procedures and governance changes
    def governance_changes():
        new = json.loads(json.dumps(CHARTER))
        new["procedures"]["labor"] = {"type": "weighted", "weights": "contribution", "electorate": "members",
                                      "window_s": 3 * DAY}
        new["rules"][1] = {"match": {"kind": "tool", "tool": "bank.transfer", "min_amount": 200_000},
                           "procedure": "labor"}
        r = g.ok("ada", "inst.propose", org=org, action={"kind": "amend", "charter": new})
        d1 = g.dec(r["decision"])
        open_after_one = d1["status"] == "open"                   # a supermajority of the admins is needed
        g.ok("lea", "inst.vote", decision=r["decision"], choice="yes")
        v = st.tables["charters"][org]["version"]
        again = g.ex("ada", "inst.adopt", org=org, charter=CHARTER)
        r2 = g.ok("ada", "inst.propose", org=org, action={"kind": "tool", "tool": "bank.transfer", "args": {
            "from_account": acct, "to": s["team"][0], "amount": 250_000, "memo": "bonus"}})
        d2 = g.dec(r2["decision"])
        el = dict(d2["electorate"])
        labor_weighted = d2["proc_name"] == "labor" and all(p in s["team"] or p in (g.p["ada"], g.p["lea"])
                                                            for p in el) and el and g.p["ada"] not in el
        s["bonus"] = r2["decision"]
        return (open_after_one and g.dec(r["decision"])["status"] == "executed" and v == 2 and not again.ok
                and labor_weighted,
                f"amended by the board's supermajority (charter v{v}); a second adoption refused; the next large "
                f"transfer is decided by labour-contribution weights over {len(el)} contributors")
    g.feature("governance_changes", governance_changes)

    def decision_procedures():
        g.advance(4)                                              # the bonus vote closes
        types = {}
        for did in st.find_ids("inst_decisions", "org", org):
            d = st.tables["inst_decisions"][did]
            if d["status"] in ("executed", "rejected", "failed", "lapsed"):
                types.setdefault(d["procedure"]["type"], set()).add(d["status"])
        g.ok("ada", "inst.propose", org=org, action={"kind": "unit_create", "name": "engineering",
                                                     "head": s["team"][0]})      # consensus (admins)
        g.advance(1.1)
        for did in st.find_ids("inst_decisions", "org", org):
            d = st.tables["inst_decisions"][did]
            if d["status"] != "open":
                types.setdefault(d["procedure"]["type"], set()).add(d["status"])
        bonus = g.dec(s["bonus"])
        sim_ballots = sum(1 for b in st.find("inst_ballots", "decision", s["bonus"]) if b["voter"] in s["team"])
        return (len(types) >= 5 and sim_ballots >= 1,
                f"procedure types that decided: {sorted(types)}; the labour-weighted bonus vote was "
                f"{bonus['status']} with {sim_ballots} ballots by simulated workers")
    g.feature("decision_procedures", decision_procedures)

    # ---------------------------------------------------------------- cooperative ventures
    def cooperative_ventures():
        beta, b_acct = OW.found_company(w, g.p["ben"], "beta", 10_000_000)
        s["beta"] = beta
        prj = g.ok("ada", "inst.project_create", title="joint platform", as_org=org, partners=[beta], lead="ada",
                   procedure={"type": "weighted", "weights": "contribution", "window_s": 2 * DAY})["project"]
        g.ok("ben", "inst.project_join", project=prj, as_org=beta)
        g.ok("ada", "inst.project_fund", project=prj, account=acct, amount=600_000)
        g.ok("ben", "inst.project_fund", project=prj, account=b_acct, amount=400_000)
        t = g.ok("ada", "inst.task_add", project=prj, title="integration", effort_hours=10)["task"]
        r = g.ok("ben", "inst.propose", org=prj, action={"kind": "allocate", "task": t, "amount": 100_000})
        g.ok("ben", "inst.vote", decision=r["decision"], choice="yes", as_org=beta)
        beta_alone = g.dec(r["decision"])["status"]               # 40% of the contributions
        g.ok("ada", "inst.vote", decision=r["decision"], choice="yes", as_org=org)
        alloc = g.dec(r["decision"])["status"]
        a0, b0 = OW.balance(w, org), OW.balance(w, beta)
        r = g.ok("ada", "inst.propose", org=prj, action={"kind": "project_close"})
        g.ok("ada", "inst.vote", decision=r["decision"], choice="yes", as_org=org)
        closed = st.tables["projects"][prj]["status"] == "closed"
        ok = beta_alone == "open" and alloc == "executed" and closed and OW.balance(w, org) == a0 + 600_000 and \
            OW.balance(w, beta) == b0 + 400_000
        return ok, "two companies pooled 1M; allocation needed contribution-weighted majority; pool returned 3:2"
    g.feature("cooperative_ventures", cooperative_ventures)

    # ---------------------------------------------------------------- restructuring (units, spin-off, merger)
    def restructuring():
        r = g.ok("ada", "inst.propose", org=org, action={"kind": "unit_create", "name": "studio", "head": "p:lea",
                                                        "budget_cents_month": 50_000})
        g.ok("lea", "inst.vote", decision=r["decision"], choice="yes")          # consensus of the admins
        unit = g.dec(r["decision"])["result"]["unit"]
        g.ok("ada", "inst.propose", org=org, action={"kind": "unit_assign", "unit": unit, "person": s["team"][-1]})
        r = g.ok("lea", "inst.propose", org=org, action={"kind": "spin_off", "unit": unit, "handle": "acme-studio",
                                                        "name": "Acme Studio", "budget_cents": 1_000_000,
                                                        "parent_stake": 0.25})
        g.ok("ada", "inst.vote", decision=r["decision"], choice="yes")
        d = g.dec(r["decision"])
        new = (d["result"] or {}).get("org")
        s["studio"] = new
        spun = d["status"] == "executed" and w.modules["identity"].role(st, new, g.p["lea"]) == "owner" and \
            OW.balance(w, new) == 1_000_000 and st.tables["shares"][f"{new}|{org}"]["n"] == 250_000
        # beta merges into acme: both organisations decide
        g.ok("ben", "inst.propose", org=s["beta"], action={"kind": "merge", "into": org})
        b_cash = OW.balance(w, s["beta"])
        a_cash = OW.balance(w, org)
        r = g.ok("ada", "inst.propose", org=org, action={"kind": "absorb", "org": s["beta"]})
        merged = g.dec(r["decision"])["status"] == "executed" and st.tables["orgs"][s["beta"]]["status"] == "merged" \
            and OW.balance(w, org) == a_cash + b_cash and OW.balance(w, s["beta"]) == 0
        return spun and merged, f"unit spun off into {new} (funded, 25% parent stake); beta merged into acme"
    g.feature("restructuring", restructuring)

    def growth():
        members = len(st.find_ids("members", "org", org))
        units = [u for u in st.find("units", "org", org) if u["status"] == "active"]
        invites = sum(1 for i in st.tables["invites"].values() if i["org"] == org and i["person"] == g.p["ben"])
        g.ok("ben", "id.respond_invite", invite=next(k for k, i in st.scan("invites") if i["org"] == org and
                                                     i["person"] == g.p["ben"] and i["status"] == "pending"),
             accept=True)
        after = len(st.find_ids("members", "org", org))
        return (members >= 4 and units and invites == 1 and after == members + 1,
                f"members {members} -> {after} (hiring, admissions by vote, a merger invitation accepted); "
                f"{len(units)} active units")
    g.feature("growth", growth)

    def dissolution():
        studio = s["studio"]
        a0 = OW.balance(w, org)
        r = g.ok("lea", "inst.propose", org=studio, action={"kind": "dissolve"})     # no charter: owners decide
        d = g.dec(r["decision"])
        ok = d["status"] == "executed" and st.tables["orgs"][studio]["status"] == "dissolved" and \
            st.tables["companies"][studio]["status"] == "dissolved" and OW.balance(w, studio) == 0
        got = OW.balance(w, org) - a0
        return ok and got == 250_000, f"studio wound up through the company registry; parent received {got} (25%)"
    g.feature("dissolution", dissolution)

    # ================================================================ stable constraints
    def attempt(name, law, fn, unchanged):
        before = _protected_snapshot(w)
        try:
            refused, detail = fn()
        except Exception as e:  # noqa: BLE001
            refused, detail = False, f"error: {type(e).__name__}: {e}"
        after = _protected_snapshot(w)
        keep = {k: before[k] == after[k] for k in unchanged}
        inv = w.check_invariants()
        g.attempts.append({"attempt": name, "law": law, "refused": bool(refused), "protected_unchanged": all(keep.values()),
                           "invariants": not inv, "blocked": bool(refused) and all(keep.values()) and not inv,
                           "detail": str(detail)[:300]})

    def amend_with(extra):
        def fn():
            c = json.loads(json.dumps(st.tables["charters"][org]["charter"]))
            c.update(extra)
            r = g.ex("ada", "inst.propose", org=org, action={"kind": "amend", "charter": c})
            return (not r.ok, r.error)
        return fn

    def decide(action, who="ada"):
        """Propose and approve with every admin's vote (the board decides unanimously)."""
        r = g.ex(who, "inst.propose", org=org, action=action)
        if not r.ok:
            return True, r.error
        did = r.result["decision"]
        for v in ("ada", "lea"):
            if g.dec(did)["status"] == "open":
                g.ex(v, "inst.vote", decision=did, choice="yes")
        d = g.dec(did)
        return d["status"] in ("failed", "rejected"), {"status": d["status"], "result": d["result"]}

    attempt("charter overrides the overdraft law", "core accounting law",
            amend_with({"accounting": {"overdraft_cents": 10**9}}), ("overdrafts", "charters", "money_total"))
    attempt("decision mints money with an operator tool", "core accounting law",
            lambda: decide({"kind": "tool", "tool": "admin.endow", "args": {"principal": org, "amount": 10**9}}),
            ("private_money", "genesis", "money_total"))
    zed_acct = w.modules["bank"].primary_account(st, g.p["zed"])
    zed0 = OW.balance(w, g.p["zed"])
    attempt("treasurer decision takes an outsider's money", "security boundary",
            lambda: (lambda r: (r[0] and OW.balance(w, g.p["zed"]) == zed0, r[1]))(
                decide({"kind": "tool", "tool": "bank.transfer",
                        "args": {"from_account": zed_acct, "to": acct, "amount": 150_000}})),
            ("private_money", "money_total"))
    attempt("decision makes a member owner without its consent", "security boundary",
            lambda: decide({"kind": "role", "person": s["team"][0], "role": "owner"}), ("owners",))
    attempt("decision grants itself authority over an outsider's resources", "security boundary",
            lambda: decide({"kind": "tool", "tool": "id.grant",
                            "args": {"grantee": org, "cap": "bank.spend", "owner": g.p["zed"]}}),
            ("grants", "hook_ids"))
    attempt("charter rewrites how its revenue is evaluated", "hidden evaluation criteria",
            amend_with({"evaluation": {"independent_revenue": "all inflows"}}), ("charters", "metrics"))
    attempt("charter conditions on an evaluator-only metric", "hidden evaluation criteria",
            amend_with({"rules": [{"match": {"kind": "tool", "tool": "bank.transfer", "when": {
                "metric": "quality_margin_vs_incumbents", "args": {"org": org, "need": "kv"}, "op": ">",
                "value": 0}}, "procedure": "members"}]}), ("charters", "metrics"))
    attempt("decision writes world state directly", "world state",
            lambda: decide({"kind": "set_state", "table": "accounts", "id": acct, "balance": 10**9}),
            ("money_total", "private_money", "clock"))
    attempt("charter moves the clock", "world state", amend_with({"clock": 0}), ("clock", "charters"))

    # ================================================================ integrity of the whole run
    inv = w.check_invariants()
    bank = w.modules["bank"]
    ledger = bank.check_ledger_consistency(w)
    rep = OW.verify_reproducible(w)
    n_features = sum(1 for ok, _ in g.features.values() if ok)
    n_blocked = sum(1 for a in g.attempts if a["blocked"])
    m = {"institution_features": n_features, "features_total": len(g.features),
         "core_law_modifications_blocked": n_blocked, "core_law_attempts": len(g.attempts),
         "laws_covered": len({a["law"] for a in g.attempts if a["blocked"]}),
         "invariant_violations": len(inv) + len(ledger),
         "money_conserved": sum(a["balance"] for a in st.tables["accounts"].values()) == 0,
         "inst_escrow_cents": st.tables["accounts"][INST_ESCROW]["balance"],
         "commit_escrow_cents": st.tables["accounts"][COMMIT_ESCROW]["balance"],
         "replay_digest_equal": rep["replay_digest_equal"], "regeneration_hash_equal": rep["regeneration_hash_equal"],
         "events": rep["events"], "simulated_days": round(w.now / DAY, 2), "simulated_workers": 16,
         "decisions": len(st.tables["inst_decisions"]),
         "simulated_worker_ballots": sum(1 for b in st.tables["inst_ballots"].values()
                                         if st.get("workers", b["voter"]) and st.tables["workers"][b["voter"]]["sim"]),
         "wall_s": round(time.time() - t0, 2), "seed": SEED}
    for name, (ok, _) in g.features.items():
        m[f"feature_{name}"] = ok
    return m, g


@pytest.mark.timeout(900)
def test_SP_6_6_adaptable_institutions_within_stable_constraints():
    m, g = run_gate()
    notes = {"features": {k: v[1] for k, v in g.features.items()}, "attempts": g.attempts}
    record("SP-6.6", m, command="PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_6_6.py",
           notes=json.dumps(notes, sort_keys=True)[:20000])
    failed = {k: v[1] for k, v in g.features.items() if not v[0]}
    assert m["institution_features"] >= 10, failed
    assert m["core_law_modifications_blocked"] >= 3, g.attempts
    assert m["invariant_violations"] == 0 and m["money_conserved"]
    assert m["replay_digest_equal"] and m["regeneration_hash_equal"]
