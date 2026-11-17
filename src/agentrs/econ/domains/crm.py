"""Sales / CRM / business operations over SQLite: contact-company matching, lead qualification,
pipeline maintenance, proposal preparation, meeting scheduling with a scripted prospect.
Verification inspects the resulting CRM state. No real outreach: the prospect is simulated."""
from __future__ import annotations

import datetime as dt
import json
import random

from agentrs.econ.core import Family, Task, ToolEnv, ToolError, company, person, register, slug

SCHEMA = """
CREATE TABLE companies(id INTEGER PRIMARY KEY, name TEXT, domain TEXT, industry TEXT, employees INTEGER, country TEXT);
CREATE TABLE contacts(id INTEGER PRIMARY KEY, name TEXT, email TEXT, title TEXT, company_id INTEGER);
CREATE TABLE leads(id INTEGER PRIMARY KEY, contact_id INTEGER, source TEXT, status TEXT);
CREATE TABLE deals(id INTEGER PRIMARY KEY, company_id INTEGER, name TEXT, stage TEXT, amount REAL, close_date TEXT, owner TEXT);
CREATE TABLE activities(id INTEGER PRIMARY KEY, deal_id INTEGER, date TEXT, type TEXT, note TEXT);
CREATE TABLE products(sku TEXT PRIMARY KEY, name TEXT, unit_price REAL);
CREATE TABLE proposals(id INTEGER PRIMARY KEY AUTOINCREMENT, deal_id INTEGER, discount_pct REAL, total REAL);
CREATE TABLE proposal_lines(proposal_id INTEGER, sku TEXT, qty INTEGER, unit_price REAL);
CREATE TABLE calendar(owner TEXT, start TEXT, end TEXT, title TEXT);
CREATE TABLE meetings(id INTEGER PRIMARY KEY AUTOINCREMENT, deal_id INTEGER, start TEXT, end TEXT);
"""
INDUSTRIES = ["software", "healthcare", "logistics", "retail", "manufacturing", "finance", "education", "energy"]
TITLES_SENIOR = ["VP of Operations", "Director of IT", "Head of Procurement", "Chief Technology Officer", "VP Sales"]
TITLES_JUNIOR = ["Analyst", "Intern", "Coordinator", "Associate", "Assistant"]
PRODUCTS = [("SKU-CRM", "CRM Seat", 49.0), ("SKU-ANL", "Analytics Add-on", 19.0), ("SKU-API", "API Pack", 299.0),
            ("SKU-ONB", "Onboarding Package", 1500.0), ("SKU-SUP", "Premium Support", 899.0), ("SKU-STO", "Storage 1TB", 120.0)]


class CRMEnv(ToolEnv):
    domain = "crm"
    role = "a sales-operations / CRM agent"

    def setup(self):
        p = self.task.private
        db = self.new_db(SCHEMA)
        for table, rows in p["tables"].items():
            for r in rows:
                db.execute(f"INSERT INTO {table}({','.join(r)}) VALUES ({','.join('?' * len(r))})", list(r.values()))
        self.rules = list(p.get("rules", []))
        self.add_sql_tool()

        @self.tool("update_contact", "update_contact(contact_id: int, company_id: int) — link a contact to a company", writes=True)
        def update_contact(contact_id, company_id):
            if not db.execute("SELECT 1 FROM contacts WHERE id=?", (int(contact_id),)).fetchone():
                raise ToolError("unknown contact")
            if not db.execute("SELECT 1 FROM companies WHERE id=?", (int(company_id),)).fetchone():
                raise ToolError("unknown company")
            db.execute("UPDATE contacts SET company_id=? WHERE id=?", (int(company_id), int(contact_id)))
            return "contact updated"

        @self.tool("set_lead_status", "set_lead_status(lead_id: int, status: 'qualified'|'disqualified') — qualify a lead",
                   writes=True)
        def set_lead_status(lead_id, status):
            if status not in ("qualified", "disqualified"):
                raise ToolError("status must be 'qualified' or 'disqualified'")
            if not db.execute("SELECT 1 FROM leads WHERE id=?", (int(lead_id),)).fetchone():
                raise ToolError("unknown lead")
            db.execute("UPDATE leads SET status=? WHERE id=?", (status, int(lead_id)))
            return "lead updated"

        @self.tool("set_deal_stage", "set_deal_stage(deal_id: int, stage: str) — move a deal to another pipeline stage",
                   writes=True)
        def set_deal_stage(deal_id, stage):
            if stage not in ("prospecting", "proposal", "negotiation", "closed_won", "closed_lost"):
                raise ToolError("invalid stage")
            if not db.execute("SELECT 1 FROM deals WHERE id=?", (int(deal_id),)).fetchone():
                raise ToolError("unknown deal")
            db.execute("UPDATE deals SET stage=? WHERE id=?", (stage, int(deal_id)))
            return "deal updated"

        @self.tool("create_proposal", "create_proposal(deal_id: int, lines: list[{sku, qty}], discount_pct: float) — "
                   "create a priced proposal", writes=True)
        def create_proposal(deal_id, lines, discount_pct):
            if not db.execute("SELECT 1 FROM deals WHERE id=?", (int(deal_id),)).fetchone():
                raise ToolError("unknown deal")
            if not isinstance(lines, list) or not lines:
                raise ToolError("lines must be a non-empty list of {sku, qty}")
            priced = []
            for ln in lines:
                if not isinstance(ln, dict) or "sku" not in ln or "qty" not in ln:
                    raise ToolError("each line needs sku and qty")
                pr = db.execute("SELECT unit_price FROM products WHERE sku=?", (ln["sku"],)).fetchone()
                if not pr:
                    raise ToolError(f"unknown sku {ln['sku']}")
                priced.append((ln["sku"], int(ln["qty"]), pr[0]))
            gross = sum(q * u for _, q, u in priced)
            total = round(gross * (1 - float(discount_pct) / 100), 2)
            cur = db.execute("INSERT INTO proposals(deal_id, discount_pct, total) VALUES (?,?,?)",
                             (int(deal_id), float(discount_pct), total))
            for sku, q, u in priced:
                db.execute("INSERT INTO proposal_lines VALUES (?,?,?,?)", (cur.lastrowid, sku, q, u))
            return f"proposal {cur.lastrowid} created: total ${total:.2f}"

        @self.tool("send_message", "send_message(text: str) — email the prospect and read their reply")
        def send_message(text):
            return "Prospect: " + p.get("prospect_reply", "Thanks, no further input from my side.")

        @self.tool("book_meeting", "book_meeting(deal_id: int, start: 'YYYY-MM-DD HH:MM') — book a 1-hour meeting",
                   writes=True)
        def book_meeting(deal_id, start):
            try:
                s = dt.datetime.strptime(str(start), "%Y-%m-%d %H:%M")
            except ValueError:
                raise ToolError("start must be 'YYYY-MM-DD HH:MM'")
            e = s + dt.timedelta(hours=1)
            db.execute("INSERT INTO meetings(deal_id,start,end) VALUES (?,?,?)",
                       (int(deal_id), s.strftime("%Y-%m-%d %H:%M"), e.strftime("%Y-%m-%d %H:%M")))
            return f"meeting booked {s:%Y-%m-%d %H:%M}-{e:%H:%M}"

        @self.tool("done", "done() — finish after all required CRM updates (terminal)", terminal=True)
        def done():
            return "finished"

    def initial_observation(self):
        return "CRM database ready. Use sql(...) to inspect it."

    def verify(self):
        for q, want in self.task.private["expect"]:
            got = sorted(tuple(r) for r in self.db.execute(q).fetchall())
            if got != sorted(tuple(w) for w in want):
                return False, {"query": q, "got": got[:6], "want": sorted(tuple(w) for w in want)[:6]}
        return True, {}


def _rng(f, split, seed):
    return random.Random(f"crm:{f}:{split}:{seed}")


def _companies(rng, split, n):
    out, names = [], set()
    while len(out) < n:
        nm = company(rng, split)
        if nm in names:
            continue
        names.add(nm)
        out.append({"id": 10 + len(out) * 3 + rng.randint(0, 2), "name": nm, "domain": slug(nm) + rng.choice([".com", ".io", ".co"]),
                    "industry": rng.choice(INDUSTRIES), "employees": rng.choice([12, 40, 85, 150, 320, 700, 2400, 9000]),
                    "country": rng.choice(["US", "DE", "UK", "CA", "JP"])})
    return out


def _contact(rng, split, cid, comp, title=None):
    nm = person(rng, split)
    first, last = nm.lower().split()
    dom = comp["domain"] if comp else rng.choice(["gmail.com", "outlook.com"])
    return {"id": cid, "name": nm, "email": f"{first}.{last}@{dom}", "title": title or rng.choice(TITLES_SENIOR + TITLES_JUNIOR),
            "company_id": None}


def make_contact_matching(split, seed):
    rng = _rng("cm", split, seed)
    comps = _companies(rng, split, 5)
    contacts, expect = [], []
    for k in range(rng.randint(3, 5)):
        c = rng.choice(comps)
        ct = _contact(rng, split, 200 + k * 2, c)
        if rng.random() < 0.3:  # e-mail on a subdomain of the company's domain
            u, d = ct["email"].split("@")
            ct["email"] = f"{u}@{rng.choice(['eu', 'mail', 'corp'])}.{d}"
        contacts.append(ct)
        expect.append((ct["id"], c["id"]))
    linked = _contact(rng, split, 299, comps[0])
    linked["company_id"] = comps[0]["id"]
    contacts.append(linked)
    expect.append((299, comps[0]["id"]))
    goal = ("Link every contact whose company_id is NULL to the company whose domain matches the contact's e-mail "
            "domain (an e-mail on a subdomain such as eu.<domain> belongs to <domain>). Then call done().")
    plan = ['{"tool": "sql", "args": {"query": "SELECT id, email FROM contacts WHERE company_id IS NULL"}}',
            '{"tool": "sql", "args": {"query": "SELECT id, name, domain FROM companies"}}']
    plan += [json.dumps({"tool": "update_contact", "args": {"contact_id": a, "company_id": b}}) for a, b in expect[:-1]]
    plan += ['{"tool": "done", "args": {}}']
    return Task("crm", "contact_matching", split, seed, goal,
                {"tables": {"companies": comps, "contacts": contacts}, "plan": plan,
                 "expect": [("SELECT id, company_id FROM contacts", expect)]}, max_steps=12)


def make_lead_qualification(split, seed):
    rng = _rng("lq", split, seed)
    comps = _companies(rng, split, 5)
    min_emp = rng.choice([50, 100, 250])
    inds = rng.sample(INDUSTRIES, 3)
    keys = ["VP", "Director", "Head", "Chief"]
    rules = [f"A lead is 'qualified' only if ALL hold: its contact's company has at least {min_emp} employees; the "
             f"company's industry is one of {', '.join(inds)}; the contact's title contains one of {', '.join(keys)}. "
             "Otherwise it is 'disqualified'."]
    for c in comps:  # decide every company's industry BEFORE computing any expectation
        if rng.random() < 0.5:
            c["industry"] = rng.choice(inds)
    contacts, leads, expect = [], [], []
    for k in range(rng.randint(3, 6)):
        c = rng.choice(comps)
        ct = _contact(rng, split, 400 + k, c)
        ct["company_id"] = c["id"]
        contacts.append(ct)
        ok = c["employees"] >= min_emp and c["industry"] in inds and any(w in ct["title"] for w in keys)
        leads.append({"id": 900 + k, "contact_id": ct["id"], "source": rng.choice(["webinar", "website", "referral"]),
                      "status": "new"})
        expect.append((900 + k, "qualified" if ok else "disqualified"))
    goal = "Qualify every lead with status 'new' according to the qualification rule, then call done()."
    plan = ['{"tool": "sql", "args": {"query": "SELECT l.id, ct.title, co.employees, co.industry FROM leads l JOIN '
            'contacts ct ON ct.id = l.contact_id JOIN companies co ON co.id = ct.company_id WHERE l.status = \'new\'"}}']
    plan += [json.dumps({"tool": "set_lead_status", "args": {"lead_id": a, "status": b}}) for a, b in expect]
    plan += ['{"tool": "done", "args": {}}']
    return Task("crm", "lead_qualification", split, seed, goal,
                {"tables": {"companies": comps, "contacts": contacts, "leads": leads}, "rules": rules, "plan": plan,
                 "expect": [("SELECT id, status FROM leads", expect)]}, max_steps=12)


def make_pipeline_update(split, seed):
    rng = _rng("pu", split, seed)
    comps = _companies(rng, split, 4)
    today = dt.date(2026, 6, 1) + dt.timedelta(days=rng.randint(0, 60))
    deals, acts, expect = [], [], []
    aid = 1
    for k in range(rng.randint(4, 6)):
        did = 50 + k
        stage = rng.choice(["prospecting", "proposal", "negotiation"])
        case = rng.choice(["won", "stale", "keep", "keep"])
        close = today + dt.timedelta(days=rng.randint(5, 60)) if case != "stale" else today - dt.timedelta(days=rng.randint(1, 30))
        if case == "keep" and rng.random() < 0.4:  # past close date but recently active: keep
            close = today - dt.timedelta(days=rng.randint(1, 20))
        deals.append({"id": did, "company_id": rng.choice(comps)["id"], "name": f"Deal {did}", "stage": stage,
                      "amount": float(rng.randint(5, 90) * 1000), "close_date": close.isoformat(), "owner": "you"})
        if case == "won":
            note = rng.choice(["Contract signed by customer", "customer returned the signed contract"])
            last = today - dt.timedelta(days=rng.randint(0, 10))
        elif case == "stale":
            note = rng.choice(["left voicemail", "sent follow-up email", "intro call"])
            last = today - dt.timedelta(days=rng.randint(31, 90))
        else:
            note = rng.choice(["demo scheduled", "pricing discussion", "sent follow-up email", "security review"])
            last = today - dt.timedelta(days=rng.randint(0, 25))
        acts.append({"id": aid, "deal_id": did, "date": last.isoformat(), "type": "note", "note": note})
        aid += 1
        new_stage = "closed_won" if case == "won" else ("closed_lost" if case == "stale" else stage)
        expect.append((did, new_stage))
    rules = ["A deal whose most recent activity note says the contract was signed moves to 'closed_won'.",
             f"Otherwise, a deal whose close_date is before today ({today.isoformat()}) AND whose most recent "
             "activity is more than 30 days old moves to 'closed_lost'.",
             "All other deals keep their stage."]
    goal = "Update the sales pipeline according to the pipeline rules, then call done()."
    plan = ['{"tool": "sql", "args": {"query": "SELECT d.id, d.stage, d.close_date, a.date, a.note FROM deals d JOIN '
            'activities a ON a.deal_id = d.id"}}']
    plan += [json.dumps({"tool": "set_deal_stage", "args": {"deal_id": a, "stage": b}})
             for (a, b), d in zip(expect, deals) if b != d["stage"]]
    plan += ['{"tool": "done", "args": {}}']
    return Task("crm", "pipeline_update", split, seed, goal,
                {"tables": {"companies": comps, "deals": deals, "activities": acts}, "rules": rules, "plan": plan,
                 "expect": [("SELECT id, stage FROM deals", expect)]}, max_steps=10)


def make_proposal(split, seed):
    rng = _rng("pp", split, seed)
    comps = _companies(rng, split, 3)
    prods = [{"sku": s, "name": n, "unit_price": p} for s, n, p in PRODUCTS]
    chosen = rng.sample(PRODUCTS, rng.randint(2, 3))
    qty = {s: (rng.choice([10, 25, 40, 80, 150, 300]) if s in ("SKU-CRM", "SKU-ANL") else rng.randint(1, 4)) for s, _, _ in chosen}
    gross = sum(qty[s] * p for s, _, p in chosen)
    disc = 10.0 if gross > 20000 else (5.0 if gross > 8000 else 0.0)
    total = round(gross * (1 - disc / 100), 2)
    did = 77
    deals = [{"id": did, "company_id": comps[0]["id"], "name": f"{comps[0]['name']} expansion", "stage": "proposal",
              "amount": 0.0, "close_date": "2026-09-30", "owner": "you"}]
    req = ", ".join(f"{qty[s]} x {n}" for s, n, _ in chosen)
    acts = [{"id": 1, "deal_id": did, "date": "2026-06-02", "type": "call", "note": f"Customer requests a quote for: {req}."}]
    rules = ["List prices are in the products table.",
             "Volume discount on the whole proposal: 10% if the list-price total exceeds $20,000; 5% if it exceeds "
             "$8,000; otherwise 0%."]
    goal = (f"Prepare the proposal for deal {did}: read the customer's request in the deal's activities, then create "
            "one proposal with the requested products and quantities and the correct discount_pct. Then call done().")
    plan = [json.dumps({"tool": "sql", "args": {"query": f"SELECT note FROM activities WHERE deal_id = {did}"}}),
            '{"tool": "sql", "args": {"query": "SELECT sku, name, unit_price FROM products"}}',
            json.dumps({"tool": "create_proposal", "args": {"deal_id": did, "lines": [{"sku": s, "qty": qty[s]} for s, _, _ in chosen],
                                                            "discount_pct": disc}}),
            '{"tool": "done", "args": {}}']
    return Task("crm", "proposal_prep", split, seed, goal,
                {"tables": {"companies": comps, "deals": deals, "activities": acts, "products": prods}, "rules": rules,
                 "plan": plan,
                 "expect": [("SELECT deal_id, discount_pct, total FROM proposals", [(did, disc, total)]),
                            ("SELECT sku, qty FROM proposal_lines", [(s, qty[s]) for s, _, _ in chosen])]}, max_steps=8)


def make_meeting(split, seed):
    rng = _rng("ms", split, seed)
    comps = _companies(rng, split, 2)
    day0 = dt.date(2026, 7, 6) + dt.timedelta(weeks=rng.randint(0, 8))  # a Monday
    did = 31
    deals = [{"id": did, "company_id": comps[0]["id"], "name": "Pilot", "stage": "negotiation", "amount": 40000.0,
              "close_date": "2026-10-01", "owner": "you"}]
    busy = []
    for d in range(3):
        day = day0 + dt.timedelta(days=d)
        for h in rng.sample(range(9, 17), rng.randint(3, 6)):
            busy.append({"owner": "you", "start": f"{day} {h:02d}:00", "end": f"{day} {h + 1:02d}:00", "title": "busy"})
    avail_days = rng.sample(range(3), 2)
    windows = []
    for d in sorted(avail_days):
        day = day0 + dt.timedelta(days=d)
        a = rng.randint(9, 14)
        windows.append((day, a, min(17, a + rng.randint(2, 4))))
    best = None
    for day, a, b in windows:
        for h in range(a, b):
            if not any(x["start"] == f"{day} {h:02d}:00" for x in busy):
                best = f"{day} {h:02d}:00"
                break
        if best:
            break
    if best is None:  # free the last hour of the first window
        day, a, b = windows[0]
        busy = [x for x in busy if x["start"] != f"{day} {b - 1:02d}:00"]
        best = f"{day} {b - 1:02d}:00"
    reply = "I'm available " + "; ".join(f"on {d.strftime('%A %Y-%m-%d')} from {a}:00 to {b}:00" for d, a, b in windows) + \
            ". Any full hour inside those windows works."
    goal = (f"Book a 1-hour meeting for deal {did} with the prospect at the EARLIEST time that is inside the prospect's "
            "availability (ask them with send_message) and does not overlap any entry of your calendar (calendar "
            "table, owner 'you'). Meetings start on the hour. Then call done().")
    plan = ['{"tool": "send_message", "args": {"text": "Could you share your availability for a one-hour meeting?"}}',
            '{"tool": "sql", "args": {"query": "SELECT start, end FROM calendar WHERE owner = \'you\' ORDER BY start"}}',
            json.dumps({"tool": "book_meeting", "args": {"deal_id": did, "start": best}}), '{"tool": "done", "args": {}}']
    return Task("crm", "meeting_scheduling", split, seed, goal,
                {"tables": {"companies": comps, "deals": deals, "calendar": busy}, "prospect_reply": reply, "plan": plan,
                 "expect": [("SELECT deal_id, start FROM meetings", [(did, best)])]}, max_steps=8)


def oracle(task: Task) -> list[str]:
    return list(task.private["plan"])


for _n, _f, _h in [("contact_matching", make_contact_matching, False), ("lead_qualification", make_lead_qualification, False),
                   ("pipeline_update", make_pipeline_update, False), ("proposal_prep", make_proposal, False),
                   ("meeting_scheduling", make_meeting, True)]:
    register(Family("crm", _n, _f, CRMEnv, oracle, heldout=_h))
