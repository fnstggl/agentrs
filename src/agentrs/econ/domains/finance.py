"""Finance & accounting over a real SQLite ledger: payment matching, expense categorisation,
invoice discrepancy detection, balance checks, budget variance. Verification is numerical and
state-based; explanations earn nothing."""
from __future__ import annotations

import datetime as dt
import json
import random

from agentrs.econ.core import Family, Task, ToolEnv, ToolError, company, register

SCHEMA = """
CREATE TABLE vendors(id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE invoices(id INTEGER PRIMARY KEY, vendor_id INTEGER, invoice_no TEXT, amount REAL, issue_date TEXT, status TEXT, flagged INTEGER DEFAULT 0, flag_reason TEXT);
CREATE TABLE invoice_lines(invoice_id INTEGER, description TEXT, qty INTEGER, unit_price REAL);
CREATE TABLE payments(id INTEGER PRIMARY KEY, vendor_id INTEGER, amount REAL, paid_date TEXT, memo TEXT, invoice_id INTEGER);
CREATE TABLE expenses(id INTEGER PRIMARY KEY, date TEXT, merchant TEXT, description TEXT, amount REAL, department TEXT, category TEXT);
CREATE TABLE ledger(id INTEGER PRIMARY KEY, account TEXT, date TEXT, description TEXT, debit REAL, credit REAL);
CREATE TABLE budgets(department TEXT, category TEXT, quarter TEXT, amount REAL);
"""
CATS = {"Travel": ["airfare", "hotel", "taxi", "train ticket"], "Software": ["saas subscription", "license renewal", "cloud hosting"],
        "Meals": ["team lunch", "client dinner", "catering"], "Office": ["printer paper", "desk chairs", "stationery"],
        "Marketing": ["ad campaign", "trade show booth", "sponsored post"]}
MERCHANTS = {"Travel": ["SkyAir", "InnStay", "CityCab", "RailCo"], "Software": ["CloudBase", "DevTools Inc", "LicenseHub"],
             "Meals": ["Bistro 21", "FoodBox", "Grill House"], "Office": ["PaperMart", "OfficeHub", "DeskWorld"],
             "Marketing": ["AdReach", "EventPro", "SocialBoost"]}


class FinanceEnv(ToolEnv):
    domain = "finance"
    role = "an accounts / finance operations agent"

    def setup(self):
        p = self.task.private
        db = self.new_db(SCHEMA)
        for table, rows in p["tables"].items():
            for r in rows:
                db.execute(f"INSERT INTO {table}({','.join(r)}) VALUES ({','.join('?' * len(r))})", list(r.values()))
        self.rules = list(p.get("rules", []))
        self.add_sql_tool()
        self.answer = None

        @self.tool("match_payment", "match_payment(payment_id: int, invoice_id: int) — record that a payment settles an invoice",
                   writes=True)
        def match_payment(payment_id, invoice_id):
            pay = db.execute("SELECT id FROM payments WHERE id=?", (int(payment_id),)).fetchone()
            inv = db.execute("SELECT id FROM invoices WHERE id=?", (int(invoice_id),)).fetchone()
            if not pay or not inv:
                raise ToolError("unknown payment or invoice id")
            db.execute("UPDATE payments SET invoice_id=? WHERE id=?", (int(invoice_id), int(payment_id)))
            db.execute("UPDATE invoices SET status='paid' WHERE id=?", (int(invoice_id),))
            return f"payment {payment_id} matched to invoice {invoice_id}"

        @self.tool("set_category", "set_category(expense_id: int, category: str) — categorise an expense", writes=True)
        def set_category(expense_id, category):
            if not db.execute("SELECT id FROM expenses WHERE id=?", (int(expense_id),)).fetchone():
                raise ToolError("unknown expense id")
            db.execute("UPDATE expenses SET category=? WHERE id=?", (str(category), int(expense_id)))
            return f"expense {expense_id} -> {category}"

        @self.tool("flag_invoice", "flag_invoice(invoice_id: int, reason: str) — flag an invoice for review", writes=True)
        def flag_invoice(invoice_id, reason):
            if not db.execute("SELECT id FROM invoices WHERE id=?", (int(invoice_id),)).fetchone():
                raise ToolError("unknown invoice id")
            db.execute("UPDATE invoices SET flagged=1, flag_reason=? WHERE id=?", (str(reason), int(invoice_id)))
            return f"invoice {invoice_id} flagged"

        @self.tool("submit_answer", "submit_answer(answer: object) — submit the requested figure/report (terminal)",
                   terminal=True)
        def submit_answer(answer):
            self.answer = answer
            return "answer recorded"

        @self.tool("done", "done() — finish after making all required updates (terminal)", terminal=True)
        def done():
            return "finished"

    def initial_observation(self):
        return "Database ready. Use sql(...) to inspect it."

    def verify(self):
        exp = self.task.private["expect"]
        db = self.db
        kind = exp["kind"]
        if kind == "state":
            for q, want in exp["queries"]:
                got = sorted(tuple(r) for r in db.execute(q).fetchall())
                if got != sorted(tuple(w) for w in want):
                    return False, {"query": q, "got": got[:6], "want": sorted(tuple(w) for w in want)[:6]}
            return True, {}
        ans = self.answer
        if kind == "number":
            try:
                v = float(str(ans).replace("$", "").replace(",", "")) if not isinstance(ans, dict) else float(ans.get("value"))
            except (TypeError, ValueError):
                return False, {"answer": str(ans)[:80]}
            return abs(v - exp["value"]) <= 0.01, {"answer": v, "want": exp["value"]}
        if kind == "variance":
            if not isinstance(ans, dict):
                return False, {"answer": str(ans)[:80]}
            try:
                amt = float(str(ans.get("amount")).replace("$", "").replace(",", ""))
            except (TypeError, ValueError):
                return False, {"answer": ans}
            ok = str(ans.get("category", "")).strip().lower() == exp["category"].lower() and abs(amt - exp["amount"]) <= 0.01
            return ok, {"answer": ans, "want": {"category": exp["category"], "amount": exp["amount"]}}
        raise ValueError(kind)


def _rng(fam, split, seed):
    return random.Random(f"finance:{fam}:{split}:{seed}")


def _vendors(rng, split, n):
    names = set()
    while len(names) < n:
        names.add(company(rng, split))
    return [{"id": i + 1, "name": nm} for i, nm in enumerate(sorted(names))]


def make_payment_matching(split, seed):
    rng = _rng("pm", split, seed)
    vendors = _vendors(rng, split, 4)
    invoices, payments = [], []
    iid = 100
    for v in vendors:
        for _ in range(rng.randint(2, 3)):
            iid += rng.randint(1, 7)
            invoices.append({"id": iid, "vendor_id": v["id"], "invoice_no": f"INV-{rng.randint(1000, 9999)}",
                             "amount": round(rng.uniform(200, 9000), 2),
                             "issue_date": (dt.date(2026, 3, 1) + dt.timedelta(days=rng.randint(0, 60))).isoformat(),
                             "status": "open"})
    targets = rng.sample(invoices, rng.randint(3, 5))
    pid = 500
    expect = []
    for inv in targets:
        pid += rng.randint(1, 5)
        memo = rng.choice([f"Payment {inv['invoice_no']}", f"ref {inv['invoice_no']}", "wire transfer", "ACH payment"])
        payments.append({"id": pid, "vendor_id": inv["vendor_id"], "amount": inv["amount"],
                         "paid_date": (dt.date.fromisoformat(inv["issue_date"]) + dt.timedelta(days=rng.randint(5, 30))).isoformat(),
                         "memo": memo, "invoice_id": None})
        expect.append((pid, inv["id"]))
    # decoy: same amount at another vendor (memo-less payments must be matched on vendor + amount)
    decoy = rng.choice(targets)
    other = rng.choice([v for v in vendors if v["id"] != decoy["vendor_id"]])
    invoices.append({"id": iid + 11, "vendor_id": other["id"], "invoice_no": f"INV-{rng.randint(1000, 9999)}",
                     "amount": decoy["amount"], "issue_date": decoy["issue_date"], "status": "open"})
    goal = ("Reconcile accounts payable: every payment with invoice_id NULL settles exactly one open invoice of the "
            "SAME vendor with the SAME amount (the memo may contain the invoice number). Match each such payment, "
            "then call done().")
    plan = ['{"tool": "sql", "args": {"query": "SELECT id, vendor_id, amount, memo FROM payments WHERE invoice_id IS NULL"}}',
            '{"tool": "sql", "args": {"query": "SELECT id, vendor_id, invoice_no, amount FROM invoices WHERE status = \'open\'"}}']
    plan += [json.dumps({"tool": "match_payment", "args": {"payment_id": p, "invoice_id": i}}) for p, i in expect]
    plan += ['{"tool": "done", "args": {}}']
    return Task("finance", "payment_matching", split, seed, goal,
                {"tables": {"vendors": vendors, "invoices": invoices, "payments": payments},
                 "expect": {"kind": "state", "queries": [("SELECT id, invoice_id FROM payments", expect)]}, "plan": plan},
                max_steps=12)


def make_expense_categorization(split, seed):
    rng = _rng("ec", split, seed)
    cats = rng.sample(list(CATS), 4)
    rules = [f"Expenses whose description mentions {', '.join(repr(k) for k in CATS[c])} are '{c}'." for c in cats]
    rows, expect = [], []
    for i in range(rng.randint(4, 7)):
        c = rng.choice(cats)
        eid = 300 + i * 5 + rng.randint(0, 4)
        rows.append({"id": eid, "date": f"2026-04-{rng.randint(1, 28):02d}", "merchant": rng.choice(MERCHANTS[c]),
                     "description": f"{rng.choice(CATS[c])} for {rng.choice(['sales', 'engineering', 'ops', 'hr'])} team",
                     "amount": round(rng.uniform(15, 2500), 2), "department": rng.choice(["Sales", "Eng", "Ops"]),
                     "category": None})
        expect.append((eid, c))
    goal = "Categorise every expense whose category is NULL using the categorisation rules, then call done()."
    plan = ['{"tool": "sql", "args": {"query": "SELECT id, description FROM expenses WHERE category IS NULL"}}']
    plan += [json.dumps({"tool": "set_category", "args": {"expense_id": e, "category": c}}) for e, c in expect]
    plan += ['{"tool": "done", "args": {}}']
    return Task("finance", "expense_categorization", split, seed, goal,
                {"tables": {"expenses": rows}, "rules": rules,
                 "expect": {"kind": "state", "queries": [("SELECT id, category FROM expenses", expect)]}, "plan": plan},
                max_steps=12)


def make_discrepancy(split, seed):
    rng = _rng("dd", split, seed)
    vendors = _vendors(rng, split, 3)
    invoices, lines, bad = [], [], []
    for k in range(rng.randint(5, 7)):
        iid = 700 + k * 3 + rng.randint(0, 2)
        ls = [(rng.choice(["consulting hours", "widgets", "support plan", "licenses", "shipping"]), rng.randint(1, 20),
               round(rng.uniform(5, 400), 2)) for _ in range(rng.randint(1, 3))]
        total = round(sum(q * u for _, q, u in ls), 2)
        if rng.random() < 0.35:
            total = round(total + rng.choice([-1, 1]) * rng.choice([10, 25, 99.5, 100, 0.5 * total]), 2)
            bad.append(iid)
        invoices.append({"id": iid, "vendor_id": rng.choice(vendors)["id"], "invoice_no": f"INV-{rng.randint(1000, 9999)}",
                         "amount": total, "issue_date": f"2026-05-{rng.randint(1, 28):02d}", "status": "open"})
        lines += [{"invoice_id": iid, "description": d, "qty": q, "unit_price": u} for d, q, u in ls]
    goal = ("Audit the open invoices: flag every invoice whose amount differs from the sum of qty*unit_price over its "
            "invoice_lines by more than $0.01 (do not flag correct invoices), then call done().")
    plan = ['{"tool": "sql", "args": {"query": "SELECT i.id, i.amount, ROUND(SUM(l.qty * l.unit_price), 2) AS line_total '
            'FROM invoices i JOIN invoice_lines l ON l.invoice_id = i.id GROUP BY i.id"}}']
    plan += [json.dumps({"tool": "flag_invoice", "args": {"invoice_id": b, "reason": "amount does not match line items"}})
             for b in bad]
    plan += ['{"tool": "done", "args": {}}']
    return Task("finance", "discrepancy_detection", split, seed, goal,
                {"tables": {"vendors": vendors, "invoices": invoices, "invoice_lines": lines},
                 "expect": {"kind": "state", "queries": [("SELECT id FROM invoices WHERE flagged=1", [(b,) for b in bad])]},
                 "plan": plan}, max_steps=10)


def make_balance(split, seed):
    rng = _rng("bc", split, seed)
    acct = rng.choice(["Cash", "Operating Account", "Petty Cash", "Accounts Receivable"])
    opening = round(rng.uniform(1000, 50000), 2)
    rows = [{"id": 1, "account": acct, "date": "2026-01-01", "description": "opening balance", "debit": opening, "credit": 0.0}]
    cutoff = dt.date(2026, 3, rng.randint(10, 25))
    bal = opening
    for k in range(rng.randint(6, 11)):
        d = dt.date(2026, 1, 2) + dt.timedelta(days=rng.randint(0, 100))
        debit = round(rng.uniform(50, 6000), 2) if rng.random() < 0.5 else 0.0
        credit = 0.0 if debit else round(rng.uniform(50, 5000), 2)
        a = acct if rng.random() < 0.75 else rng.choice(["Payroll", "Inventory", "Savings"])
        rows.append({"id": k + 2, "account": a, "date": d.isoformat(), "description": rng.choice(
            ["customer receipt", "supplier payment", "bank fee", "transfer", "refund"]), "debit": debit, "credit": credit})
        if a == acct and d <= cutoff:
            bal += debit - credit
    bal = round(bal, 2)
    goal = (f"What is the balance of the '{acct}' account at the end of {cutoff.isoformat()} (inclusive)? Balance = "
            f"sum(debit) - sum(credit) over that account's ledger entries up to that date, including the opening "
            f"entry. Submit the number with submit_answer.")
    q = (f"SELECT ROUND(SUM(debit) - SUM(credit), 2) FROM ledger WHERE account = '{acct}' AND date <= '{cutoff.isoformat()}'")
    plan = [json.dumps({"tool": "sql", "args": {"query": q}}), json.dumps({"tool": "submit_answer", "args": {"answer": bal}})]
    return Task("finance", "balance_check", split, seed, goal,
                {"tables": {"ledger": rows}, "expect": {"kind": "number", "value": bal}, "plan": plan}, max_steps=8)


def make_budget_variance(split, seed):
    rng = _rng("bv", split, seed)
    dept = rng.choice(["Sales", "Engineering", "Operations", "Marketing"])
    q = rng.choice(["Q1", "Q2"])
    months = {"Q1": (1, 3), "Q2": (4, 6)}[q]
    cats = rng.sample(list(CATS), 3)
    budgets, exps = [], []
    spend = {}
    eid = 1
    for c in cats:
        b = float(rng.randint(20, 80) * 100)
        budgets.append({"department": dept, "category": c, "quarter": q, "amount": b})
        total = 0.0
        for _ in range(rng.randint(2, 4)):
            amt = round(rng.uniform(0.15, 0.55) * b, 2)
            m = rng.randint(*months)
            exps.append({"id": eid, "date": f"2026-{m:02d}-{rng.randint(1, 28):02d}", "merchant": rng.choice(MERCHANTS[c]),
                         "description": rng.choice(CATS[c]), "amount": amt, "department": dept, "category": c})
            eid += 1
            total += amt
        # out-of-quarter and other-department noise
        exps.append({"id": eid, "date": f"2026-{(months[1] % 12) + 1:02d}-03", "merchant": rng.choice(MERCHANTS[c]),
                     "description": rng.choice(CATS[c]), "amount": round(rng.uniform(100, 900), 2), "department": dept,
                     "category": c})
        eid += 1
        spend[c] = round(total - b, 2)
    worst = max(spend, key=lambda c: spend[c])
    if spend[worst] <= 0:  # ensure a real overspend exists
        exps.append({"id": eid, "date": f"2026-{months[0]:02d}-15", "merchant": rng.choice(MERCHANTS[worst]),
                     "description": rng.choice(CATS[worst]), "amount": round(-spend[worst] + rng.uniform(100, 900), 2),
                     "department": dept, "category": worst})
        spend[worst] = round(spend[worst] + exps[-1]["amount"], 2)
        worst = max(spend, key=lambda c: spend[c])
    goal = (f"For the {dept} department in {q} 2026, find the expense category with the largest overspend "
            f"(actual expenses dated within the quarter minus budget). Submit "
            f"submit_answer(answer={{\"category\": ..., \"amount\": <overspend>}}).")
    m0, m1 = months
    plan = [json.dumps({"tool": "sql", "args": {"query": (
        f"SELECT e.category, ROUND(SUM(e.amount) - b.amount, 2) AS over FROM expenses e JOIN budgets b ON "
        f"b.category = e.category AND b.department = e.department AND b.quarter = '{q}' WHERE e.department = '{dept}' "
        f"AND e.date >= '2026-{m0:02d}-01' AND e.date <= '2026-{m1:02d}-31' GROUP BY e.category ORDER BY over DESC")}}),
        json.dumps({"tool": "submit_answer", "args": {"answer": {"category": worst, "amount": spend[worst]}}})]
    return Task("finance", "budget_variance", split, seed, goal,
                {"tables": {"budgets": budgets, "expenses": exps},
                 "expect": {"kind": "variance", "category": worst, "amount": spend[worst]}, "plan": plan}, max_steps=8)


def oracle(task: Task) -> list[str]:
    return list(task.private["plan"])


for _n, _f, _h in [("payment_matching", make_payment_matching, False), ("expense_categorization", make_expense_categorization, False),
                   ("discrepancy_detection", make_discrepancy, False), ("balance_check", make_balance, False),
                   ("budget_variance", make_budget_variance, True)]:
    register(Family("finance", _n, _f, FinanceEnv, oracle, heldout=_h))
