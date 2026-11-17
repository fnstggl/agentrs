"""Research & analysis over a document collection with conflicting sources (audited annual reports
vs preliminary press releases vs news). Answers must be attributable: the verifier checks the
value numerically/exactly AND that the cited evidence includes the authoritative documents.
All grading is mechanical; there is no LLM-judged quality score in this domain."""
from __future__ import annotations

import ast
import json
import operator
import random
import re

from agentrs.econ.core import Family, Task, ToolEnv, ToolError, company, register, slug

YEARS = [2022, 2023, 2024]


class ResearchEnv(ToolEnv):
    domain = "research"
    role = "a research analyst"
    rules = ["Audited annual reports take precedence over press releases and news articles.",
             "Cite the documents your answer relies on (doc ids) in submit(evidence=[...])."]

    def setup(self):
        docs = self.task.private["docs"]
        self.docs = {d["id"]: d for d in docs}
        self.answer = None
        self.evidence: list = []

        @self.tool("search", "search(query: str) — find documents (returns ids and titles)")
        def search(query):
            words = [w for w in re.findall(r"[a-z0-9]+", str(query).lower()) if len(w) > 2]
            if not words:
                raise ToolError("empty query")
            scored = []
            for d in docs:
                text = (d["title"] + " " + d["text"]).lower()
                s = sum(1 for w in words if w in text)
                if s:
                    scored.append((-s, d["id"], d["title"]))
            scored.sort()
            return "\n".join(f"{i}: {t}" for _, i, t in scored[:8]) or "no results"

        @self.tool("read", "read(doc_id: str) — read a document")
        def read(doc_id):
            d = self.docs.get(str(doc_id))
            if not d:
                raise ToolError("unknown doc id")
            return f"[{d['id']}] {d['title']}\n{d['text']}"

        @self.tool("calculate", "calculate(expression: str) — evaluate arithmetic, e.g. '(512.4-430.1)/430.1*100'")
        def calculate(expression):
            return str(round(_safe_eval(str(expression)), 6))

        @self.tool("submit", "submit(answer: str|number, evidence: list[str]) — final answer with cited doc ids (terminal)",
                   terminal=True)
        def submit(answer, evidence):
            self.answer = answer
            self.evidence = [str(e) for e in (evidence if isinstance(evidence, list) else [evidence])]
            return "answer submitted"

    def initial_observation(self):
        return f"Document collection with {len(self.docs)} documents. Use search(...) and read(...)."

    def verify(self):
        exp = self.task.private["expect"]
        missing = [e for e in exp["evidence"] if e not in self.evidence]
        ans = self.answer
        if exp["kind"] == "number":
            try:
                v = float(str(ans).replace("%", "").replace("$", "").replace(",", "").replace("million", "").strip())
                ok_val = abs(v - exp["value"]) <= exp["tol"]
            except (TypeError, ValueError):
                ok_val = False
        elif exp["kind"] == "list":
            got = [x.strip().lower() for x in str(ans).split(",")] if not isinstance(ans, list) else [str(x).strip().lower() for x in ans]
            ok_val = got == [x.lower() for x in exp["value"]]
        else:
            ok_val = str(ans).strip().lower().rstrip(".") == str(exp["value"]).lower()
        return ok_val and not missing, {"answer": str(ans)[:80], "want": exp["value"], "missing_evidence": missing}


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(expr: str) -> float:
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            r = ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(r) > 8:
                raise ToolError("exponent too large")
            return _OPS[type(n.op)](ev(n.left), r)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ToolError("only numbers and + - * / ** ( ) are allowed")
    try:
        return float(ev(ast.parse(expr.replace(",", ""), mode="eval")))
    except ZeroDivisionError:
        raise ToolError("division by zero")
    except SyntaxError:
        raise ToolError("invalid expression")


def _world(rng, split):
    names = []
    while len(names) < 4:
        n = company(rng, split)
        if n not in names:
            names.append(n)
    data, docs = {}, []
    for n in names:
        rev = rng.uniform(80, 900)
        for y in YEARS:
            rev = round(rev * rng.uniform(0.85, 1.35), 1)
            margin = rng.uniform(-0.05, 0.3)
            data[(n, y)] = {"revenue": rev, "op_income": round(rev * margin, 1), "headcount": int(rev * rng.uniform(1.5, 4))}
    for n in names:
        for y in YEARS[1:]:
            d = data[(n, y)]
            prev = data[(n, y - 1)]
            docs.append({"id": f"AR-{slug(n)[:10]}-{y}", "title": f"{n} Annual Report {y} (audited)",
                         "text": (f"{n} reports audited results for fiscal {y}. Revenue was ${d['revenue']} million "
                                  f"(fiscal {y - 1}: ${prev['revenue']} million). Operating income was ${d['op_income']} "
                                  f"million. The company had {d['headcount']} employees at the end of {y}.")})
    for n in rng.sample(names, 3):
        y = rng.choice(YEARS[1:])
        d = data[(n, y)]
        docs.append({"id": f"PR-{slug(n)[:10]}-{y}", "title": f"{n} announces preliminary {y} results",
                     "text": (f"Preliminary, unaudited: {n} expects fiscal {y} revenue of about "
                              f"${round(d['revenue'] * rng.uniform(0.9, 1.1), 1)} million and roughly "
                              f"{int(d['headcount'] * rng.uniform(0.85, 1.15))} employees.")})
    a, b = rng.sample(names, 2)
    docs.append({"id": f"NEWS-{rng.randint(100, 999)}", "title": f"Industry roundup: {a} and {b}",
                 "text": f"Analysts compared {a} and {b}; both expanded headcount. Figures in this article are estimates."})
    rng.shuffle(docs)
    return names, data, docs


def _ar(n, y):
    return f"AR-{slug(n)[:10]}-{y}"


def _search(q):
    return json.dumps({"tool": "search", "args": {"query": q}})


def _read(i):
    return json.dumps({"tool": "read", "args": {"doc_id": i}})


def _submit(ans, ev):
    return json.dumps({"tool": "submit", "args": {"answer": ans, "evidence": ev}})


def _task(family, split, seed, goal, docs, expect, plan, max_steps=10):
    return Task("research", family, split, seed, goal, {"docs": docs, "expect": expect, "plan": plan}, max_steps=max_steps)


def make_fact(split, seed):
    rng = random.Random(f"research:fact:{split}:{seed}")
    names, data, docs = _world(rng, split)
    n, y = rng.choice(names), rng.choice(YEARS[1:])
    v = data[(n, y)]["revenue"]
    goal = f"What was {n}'s revenue in fiscal {y}, in $ millions? Submit the number and cite your evidence."
    plan = [_search(f"{n} {y} revenue"), _read(_ar(n, y)), _submit(v, [_ar(n, y)])]
    return _task("fact_extraction", split, seed, goal, docs, {"kind": "number", "value": v, "tol": 0.05, "evidence": [_ar(n, y)]}, plan)


def make_comparison(split, seed):
    rng = random.Random(f"research:cmp:{split}:{seed}")
    names, data, docs = _world(rng, split)
    a, b = rng.sample(names, 2)
    y = rng.choice(YEARS[1:])
    ma = data[(a, y)]["op_income"] / data[(a, y)]["revenue"]
    mb = data[(b, y)]["op_income"] / data[(b, y)]["revenue"]
    win = a if ma > mb else b
    goal = (f"Which company had the higher operating margin (operating income / revenue) in fiscal {y}: {a} or {b}? "
            "Submit the company name and cite your evidence.")
    plan = [_search(f"{a} {y}"), _read(_ar(a, y)), _search(f"{b} {y}"), _read(_ar(b, y)),
            json.dumps({"tool": "calculate", "args": {"expression": f"{data[(a, y)]['op_income']}/{data[(a, y)]['revenue']} - "
                                                                    f"{data[(b, y)]['op_income']}/{data[(b, y)]['revenue']}"}}),
            _submit(win, [_ar(a, y), _ar(b, y)])]
    return _task("comparison", split, seed, goal, docs, {"kind": "text", "value": win, "evidence": [_ar(a, y), _ar(b, y)]}, plan)


def make_growth(split, seed):
    rng = random.Random(f"research:growth:{split}:{seed}")
    names, data, docs = _world(rng, split)
    n = rng.choice(names)
    v0, v1 = data[(n, 2023)]["revenue"], data[(n, 2024)]["revenue"]
    g = round((v1 - v0) / v0 * 100, 1)
    goal = (f"By what percentage did {n}'s revenue grow from fiscal 2023 to fiscal 2024? Answer in percent rounded to "
            "one decimal (negative if it shrank) and cite your evidence.")
    plan = [_search(f"{n} annual report"), _read(_ar(n, 2024)), _read(_ar(n, 2023)),
            json.dumps({"tool": "calculate", "args": {"expression": f"({v1}-{v0})/{v0}*100"}}),
            _submit(g, [_ar(n, 2024), _ar(n, 2023)])]
    # 2024's report also states the 2023 figure, so citing it alone is accepted for the 2023 value too
    return _task("growth_calc", split, seed, goal, docs, {"kind": "number", "value": g, "tol": 0.1, "evidence": [_ar(n, 2024)]}, plan)


def make_ranking(split, seed):
    rng = random.Random(f"research:rank:{split}:{seed}")
    names, data, docs = _world(rng, split)
    sel = rng.sample(names, 3)
    y = rng.choice(YEARS[1:])
    order = sorted(sel, key=lambda n: -data[(n, y)]["revenue"])
    goal = (f"Rank {', '.join(sel)} by fiscal {y} revenue, highest first. Submit the names as a comma-separated list "
            "and cite your evidence.")
    plan = []
    for n in sel:
        plan += [_search(f"{n} {y}"), _read(_ar(n, y))]
    plan.append(_submit(", ".join(order), [_ar(n, y) for n in sel]))
    return _task("ranking", split, seed, goal, docs, {"kind": "list", "value": order, "evidence": [_ar(n, y) for n in sel]},
                 plan, max_steps=10)


def make_verification(split, seed):
    rng = random.Random(f"research:verify:{split}:{seed}")
    names, data, docs = _world(rng, split)
    n, y = rng.choice(names), rng.choice(YEARS[1:])
    hc = data[(n, y)]["headcount"]
    thr = int(hc * rng.choice([0.8, 0.9, 1.1, 1.25]))
    truth = "true" if hc > thr else "false"
    goal = (f"Fact-check this claim: \"{n} had more than {thr} employees at the end of {y}.\" Submit 'true' or 'false' "
            "and cite your evidence.")
    plan = [_search(f"{n} employees {y}"), _read(_ar(n, y)), _submit(truth, [_ar(n, y)])]
    return _task("fact_verification", split, seed, goal, docs, {"kind": "text", "value": truth, "evidence": [_ar(n, y)]}, plan)


def oracle(task: Task) -> list[str]:
    return list(task.private["plan"])


for _n, _f, _h in [("fact_extraction", make_fact, False), ("comparison", make_comparison, False),
                   ("growth_calc", make_growth, False), ("ranking", make_ranking, False),
                   ("fact_verification", make_verification, True)]:
    register(Family("research", _n, _f, ResearchEnv, oracle, heldout=_h))
