"""Evaluation sets for the eight class-4 capabilities (MASTER_SPEC 3.2), built from world situations.

Every item is staged in a kernel world of the 4.0 open economy (`open_digital`) where that is meaningful: documents
are created with ``docs.create`` and read with ``docs.get``, mail is sent and delivered, invoices, contracts, invites,
forum replies and failed actions are real commands, and the tool lists are the world's own catalogue. The reader's
context is rendered exactly as a `SemanticParticipant` sees it (`SemanticParticipant.header`), so the benchmark
measures the inputs participants get. Text-only situations (incentives, negotiation, strategy) are framed with world
entities; their correct answers are computed, never judged.

Splits: ``train`` (only for training task-specific heads), ``dev`` (only for choosing a scoring method per model)
and ``test`` (reported). Wherever a capability has templates, test items come from templates never used in train or
dev (``held_out_template: true``); the remaining test items are fresh instances of seen templates and are reported
separately.

Item: ``{id, cap, split, template, held_out_template, format: mc|gen, messages, prefix_messages, query,
options, answer, chance, eval}``; ``mc`` items have options and one correct answer; ``gen`` items carry what the
automatic checker needs in ``eval``.
"""
from __future__ import annotations

import json
import re

import numpy as np

from ..participants import BM25, SYSTEM_PROMPT, SemanticParticipant, tool_line, tool_text

SPLITS = ("train", "dev", "test")
NAMES = ["ana", "ben", "cara", "dev", "eli", "fay", "gus", "hana", "ivo", "jun", "kai", "lea", "milo", "nia", "omar",
         "pia", "quin", "rosa", "sam", "tara", "uma", "vik", "wes", "xena", "yara", "zeno"]
COMPANIES = ["Brightbyte", "Northwind Labs", "Copperleaf", "Bluefin Data", "Stonebridge", "Lumen Works", "Redkite",
             "Oakline", "Silverpine", "Tidewater", "Amberline", "Quartzfield"]
PRODUCTS = ["route planner", "invoice tool", "image resizer", "search index", "chat widget", "backup service",
            "translation API", "weather feed", "price tracker", "log analyzer", "booking calendar", "survey form"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
ROOMS = ["room 2", "room 5", "the library", "the main hall", "room 11", "the cafe"]
CAUSES = ["an expired certificate", "a full disk", "a bad configuration change", "a power failure in the data centre",
          "a database lock", "a memory leak", "a network cable cut", "an overloaded cache"]
FEATURES = ["offline mode", "priority support", "custom domains", "team accounts", "data export", "audit logs",
            "single sign-on", "daily backups", "an API", "dark mode"]
THINGS = ["logo", "website text", "database setup", "translation", "market survey", "server migration", "user guide",
          "photo shoot"]
TOPICS = ["cheap hosting", "pricing a new app", "finding a co-founder", "writing good documentation",
          "hiring a first developer", "reducing server costs"]


def _choice(rng, seq):
    return seq[int(rng.integers(0, len(seq)))]


def _sample(rng, seq, k):
    idx = rng.choice(len(seq), size=k, replace=False)
    return [seq[int(i)] for i in idx]


def _money(c: int) -> str:
    return f"${c / 100:,.2f}"


# ====================================================================== the staging world
class Bench:
    """A kernel world in which benchmark items are staged by real tool calls (one fresh reader per item)."""

    def __init__(self, seed: int, family: str = "open_digital"):
        import crucible
        from ...core.kernel import World
        from ...worldgen import generator as G
        crucible.load_all_modules()
        self.world = World.create(G.generate({"family": family,
                                              "overrides": {"params": {"cost_model": "deterministic"}}}, seed))
        self.n = 0
        self._tools = None

    def person(self, base: str = "r", endowment: int = 200_000) -> tuple[str, str]:
        self.n += 1
        r = self.world.system_call("admin.join", {"handle": f"{base}{self.n}x", "endowment": endowment})
        return r["principal"], r["account"]

    def do(self, actor: str, tool: str, args: dict) -> dict:
        return self.world.execute(actor, {"tool": tool, "args": args}).to_dict()

    def must(self, actor: str, tool: str, args: dict) -> dict:
        r = self.do(actor, tool, args)
        if not r["ok"]:
            raise RuntimeError(f"staging {tool} failed: {r.get('error')}")
        return r["result"]

    def tools(self, actor: str) -> list[dict]:
        if self._tools is None:
            self._tools = [t for t in self.world.available_tools(actor) if not t["name"].startswith("admin.")]
        return self._tools

    def view(self, actor: str, goal: str, since: int, actions=(), advance: int = 300) -> SemanticParticipant:
        """The participant's view: its own actions (results in memory) and the events it saw since `since`."""
        if advance:
            self.world.advance_time(self.world.now + advance)
        p = SemanticParticipant(None, goal)
        p.actor = actor
        for tool, args in actions:
            res = self.do(actor, tool, args)
            p.observe_result({"tool": tool, "args": args}, res)
        obs = self.world.observe(actor)
        evs, _ = self.world.visible_events(actor, since, limit=30)
        obs["events"] = [{"seq": e.seq, "t": e.t, "type": e.type, "actor": e.actor, "payload": e.payload}
                         for e in evs if not (e.actor == actor and e.type in ("cmd", "cmd.rejected"))]
        p.ingest(obs)
        p.memory.sort(key=lambda m: m["t"])
        return p


def _mc(cap, split, template, k, held, p_or_ctx, question, options, answer, rng, chance=None, extra=None) -> dict:
    order = [int(i) for i in rng.permutation(len(options))]
    opts = [options[i] for i in order]
    ans = order.index(answer)
    ctx = p_or_ctx.header() if isinstance(p_or_ctx, SemanticParticipant) else p_or_ctx
    user = (f"{ctx}\n\nQuestion: {question}\nOptions:\n" + "\n".join(f"- {o}" for o in opts) +
            "\nAnswer with the text of the correct option only.")
    it = {"id": f"{cap}/{split}/{template}/{k}", "cap": cap, "split": split, "template": template,
          "held_out_template": held, "format": "mc",
          "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
          "prefix_messages": 1, "query": f"{ctx}\n{question}", "options": opts, "answer": ans,
          "chance": chance if chance is not None else 1.0 / len(opts)}
    if extra:
        it.update(extra)
    return it


def _situation(actor: str, text: str, goal: str = "") -> str:
    return f"You are {actor}.\n" + (f"Goal: {goal}\n" if goal else "") + f"Situation: {text}"


# ====================================================================== 1. content understanding
def _content_doc(rng, t):
    a, b = _sample(rng, COMPANIES, 2)
    if t == "quote":
        pa, pb = [int(x) * 5 for x in _sample(rng, list(range(20, 80)), 2)]
        da, db = _sample(rng, list(range(2, 15)), 2)
        n = int(rng.integers(3, 12))
        prod = _choice(rng, PRODUCTS)
        body = (f"Hello, we asked two suppliers for {n} licences of a {prod}. {a} quoted {pa} cents per licence with "
                f"delivery in {da} days. {b} quoted {pb} cents per licence and can deliver in {db} days. Both include "
                f"one year of updates.")
        qs = [("Which supplier offers the lower price per licence?", [a if pa < pb else b, b if pa < pb else a,
                                                                      "Both quoted the same price",
                                                                      _choice(rng, [c for c in COMPANIES if c not in (a, b)])], 0),
              ("Which supplier can deliver sooner?", [a if da < db else b, b if da < db else a, "Neither gave a date",
                                                      "Both deliver on the same day"], 0),
              (f"How much would {n} licences from {a} cost in total?", [f"{n * pa} cents", f"{n * pb} cents",
                                                                         f"{pa} cents", f"{n * pa + pb} cents"], 0)]
        return f"Quote comparison for {prod}", body, qs
    if t == "meeting":
        d0, d1 = _sample(rng, DAYS[:5], 2)
        h0, h1 = _sample(rng, list(range(9, 18)), 2)
        r0, r1 = _sample(rng, ROOMS, 2)
        body = (f"Team update: the planning meeting that was set for {d0} at {h0}:00 in {r0} has been moved. It will "
                f"now take place on {d1} at {h1}:00 in {r1}. Please bring your notes.")
        qs = [("When does the planning meeting take place now?", [f"{d1} at {h1}:00", f"{d0} at {h0}:00",
                                                                  f"{d1} at {h0}:00", f"{d0} at {h1}:00"], 0),
              ("Where will the planning meeting be held?", [r1, r0] + _sample(rng, [r for r in ROOMS if r not in (r0, r1)], 2), 0)]
        return "Meeting moved", body, qs
    if t == "incident":
        svc = _choice(rng, PRODUCTS)
        c, wrong = _sample(rng, CAUSES, 2)
        h = int(rng.integers(2, 9))
        n = int(rng.integers(20, 400))
        body = (f"Incident report: our {svc} was unavailable for {h} hours on day 3. At first we suspected {wrong}, "
                f"but that was ruled out. The real cause was {c}. {n} customers were affected and we have refunded "
                f"them.")
        others = [x for x in CAUSES if x not in (c, wrong)]
        qs = [("What caused the outage?", [c, wrong] + _sample(rng, others, 2), 0),
              ("How long was the service unavailable?", [f"{h} hours", f"{h + 2} hours", f"{n} hours", "3 hours" if h != 3 else "5 hours"], 0)]
        return f"Incident report: {svc}", body, qs
    if t == "plan":
        f = _sample(rng, FEATURES, 4)
        x, y = int(rng.integers(5, 15)) * 100, int(rng.integers(20, 40)) * 100
        body = (f"Pricing: the Basic plan costs {x} cents per month and includes {f[0]} and {f[1]}. The Pro plan costs "
                f"{y} cents per month and includes everything in Basic plus {f[2]}, but it does not include {f[3]}, "
                f"which is sold separately.")
        qs = [("Which feature is NOT included in the Pro plan?", [f[3], f[0], f[1], f[2]], 0),
              ("How much does the Pro plan cost per month?", [f"{y} cents", f"{x} cents", f"{x + y} cents", f"{y - x} cents"], 0),
              (f"Does the Basic plan include {f[2]}?", ["No, only Pro includes it", "Yes", "Only if sold separately",
                                                       "The text does not mention it"], 0)]
        return "Plans and prices", body, qs
    if t == "job":
        role = _choice(rng, ["backend developer", "support engineer", "data analyst", "designer"])
        sal = int(rng.integers(30, 90)) * 10000
        req, other = _sample(rng, ["Python", "SQL", "customer support experience", "Go", "statistics", "Figma"], 2)
        day = int(rng.integers(5, 30))
        body = (f"{a} is hiring a {role}. The salary is {sal} cents per month. Applicants must know {req}; knowledge "
                f"of {other} is a plus but not required. Applications close on day {day}.")
        qs = [("What is the monthly salary?", [f"{sal} cents", f"{sal // 10} cents", f"{sal * 12} cents", f"{day} cents"], 0),
              ("Which skill is required?", [req, other, "none", role], 0),
              ("When do applications close?", [f"day {day}", f"day {day + 7}", "they never close", f"day {max(1, day - 4)}"], 0)]
        return f"Job: {role} at {a}", body, qs
    if t == "invoice":
        amt = int(rng.integers(20, 200)) * 100
        paid = int(rng.integers(1, amt // 100 - 1)) * 100
        fee = int(rng.integers(2, 10))
        day = int(rng.integers(5, 25))
        body = (f"Reminder: invoice INV-{int(rng.integers(100, 999))} over {amt} cents for the {_choice(rng, THINGS)} is due on "
                f"day {day}. We received a partial payment of {paid} cents. After the due date a late fee of {fee} "
                f"percent applies to the remaining amount.")
        qs = [("How much is still owed?", [f"{amt - paid} cents", f"{amt} cents", f"{paid} cents", f"{amt + paid} cents"], 0),
              ("What happens if the remaining amount is paid after the due date?",
               [f"A late fee of {fee} percent is added", "Nothing", "The invoice is cancelled",
                f"The partial payment of {paid} cents is refunded"], 0)]
        return "Payment reminder", body, qs
    if t == "review":
        x, y = _sample(rng, PRODUCTS, 2)
        body = (f"I tried both {x} and {y} for a month. {x} was clearly faster, but it crashed twice and lost my "
                f"data once. {y} was slower, yet it never failed. Reliability matters more to me than speed, so I will "
                f"keep using {y}.")
        qs = [("Which product will the author keep using?", [y, x, "neither", "both"], 0),
              ("Why did the author make that choice?", [f"{y} never failed", f"{y} was faster", f"{y} was cheaper",
                                                        f"{x} lost data only once"], 0)]
        return f"Review: {x} vs {y}", body, qs
    if t == "policy":
        n = int(rng.integers(7, 30))
        pct = int(rng.integers(3, 8)) * 10
        m = n + int(rng.integers(2, 20))
        body = (f"Refund policy: unused items can be returned within {n} days for a full refund. Opened items get a "
                f"{pct} percent refund within the same period. Digital goods cannot be refunded.")
        qs = [("A customer bought a digital game yesterday. Can they get a refund?",
               ["No, digital goods cannot be refunded", "Yes, a full refund", f"Yes, a {pct} percent refund",
                f"Only within {n} days"], 0),
              (f"A customer returns an unused item after {m} days. What do they get?",
               ["No refund, the period has ended", "A full refund", f"A {pct} percent refund", "A replacement item"], 0),
              ("A customer returns an opened item after 2 days. What do they get?",
               [f"A {pct} percent refund", "A full refund", "No refund", "Store credit only"], 0)]
        return "Refund policy", body, qs
    raise KeyError(t)


CONTENT_TEMPLATES = {"train": ["quote", "meeting", "incident", "job", "invoice"],
                     "test_new": ["plan", "review", "policy"]}


def build_content(bench: Bench, split: str, n: int, rng) -> list[dict]:
    out = []
    for k in range(n):
        held = split == "test" and k % 2 == 0
        t = _choice(rng, CONTENT_TEMPLATES["test_new"] if held else CONTENT_TEMPLATES["train"])
        title, body, qs = _content_doc(rng, t)
        q, opts, ans = qs[int(rng.integers(0, len(qs)))]
        author, _ = bench.person("w")
        reader, _ = bench.person("r")
        since = bench.world.seq
        doc = bench.must(author, "docs.create", {"title": title, "body": body, "visibility": "public"})["doc"]
        p = bench.view(reader, "Read the documents you are given and answer questions about them.", since,
                       actions=[("docs.get", {"doc": doc})])
        out.append(_mc("content_understanding", split, t, k, held, p, q, opts, ans, rng))
    return out


# ====================================================================== 2. situation interpretation
def _stage_situation(bench: Bench, t: str, rng) -> tuple:
    me, acct = bench.person("r", int(rng.integers(20, 200)) * 100)
    other, oacct = bench.person(_choice(rng, NAMES))
    since = bench.world.seq
    goal = "Keep track of what is happening to you and handle it sensibly."
    acts = []
    if t == "afford_invoice":
        amt = int(rng.integers(10, 300)) * 100
        bench.must(other, "bank.invoice", {"amount": amt, "payer": me, "to_account": oacct, "memo": _choice(rng, THINGS)})
        acts = [("bank.accounts", {})]
        p = bench.view(me, goal, since, acts)
        bal = p.history[-1]["result"][0]["balance"]
        ok = bal >= amt
        q = "Can you pay the invoice you received in full from your checking account right now?"
        opts = ["Yes, your balance is enough", "No, your balance is too low", "There is no invoice",
                "The invoice has already been paid"]
        return p, q, opts, 0 if ok else 1
    if t == "who_waits":
        subj = _choice(rng, ["Question about the " + x for x in THINGS])
        third, _ = bench.person(_choice(rng, NAMES))
        bench.must(other, "mail.send", {"to": [me], "subject": subj, "body": "Could you get back to me today?"})
        bench.must(third, "forum.post", {"board": "general", "title": "Hello all", "body": "Just saying hi."})
        p = bench.view(me, goal, since)
        q = "Who sent you a message that you have not answered yet?"
        opts = [other, third, me, "nobody"]
        return p, q, opts, 0
    if t == "why_failed":
        kind = _choice(rng, ["funds", "recipient", "args"])
        if kind == "funds":
            acts = [("bank.transfer", {"from_account": acct, "to": other, "amount": 10 ** 8})]
            ans = 0
        elif kind == "recipient":
            acts = [("bank.transfer", {"from_account": acct, "to": "p:nosuchperson" + str(int(rng.integers(100, 999))), "amount": 100})]
            ans = 1
        else:
            acts = [("mail.send", {"to": [other], "subject": "hi", "body": "hello", "colour": "blue"})]
            ans = 2
        p = bench.view(me, goal, since, acts)
        q = "Why did your last action fail?"
        opts = ["There was not enough money in the account", "The recipient does not exist",
                "The call had an argument the tool does not accept", "The service was down for maintenance"]
        return p, q, opts, ans
    if t == "cheapest_provider":
        acts = [("cloud.providers", {})]
        p = bench.view(me, "Host a small service as cheaply as possible.", since, acts)
        provs = p.history[-1]["result"]
        best = min(provs, key=lambda x: x["price_core_hour_cents"])
        names = [x["provider"] for x in provs]
        others = [x for x in names if x != best["provider"]]
        opts = [best["provider"]] + others[:2] + ["none of them has free capacity"]
        while len(opts) < 4:
            opts.insert(-1, f"o:{_choice(rng, NAMES)}-cloud")
        q = "Which provider charges the least per core-hour?"
        return p, q, opts[:4], 0
    if t == "forum_reply":
        title = "Question about " + _choice(rng, TOPICS)
        post = bench.must(me, "forum.post", {"board": "help", "title": title, "body": "Any advice?"})["post"]
        since = bench.world.seq
        bench.must(other, "forum.reply", {"post": post, "body": "I can help, try the docs first."})
        p = bench.view(me, goal, since)
        q = "What happened to your forum post?"
        opts = [f"{other} replied to it", "It was deleted", "It received a down-vote", "Nothing has happened to it"]
        return p, q, opts, 0
    if t == "contract_waiting":
        thing = _choice(rng, THINGS)
        amt = int(rng.integers(10, 90)) * 1000
        bench.must(other, "contracts.propose", {"parties": [other, me], "title": f"{thing} work",
                                                "clauses": [{"type": "payment", "from": other, "to": me, "amount": amt}]})
        p = bench.view(me, goal, since)
        q = f"What is {other} waiting for from you?"
        opts = [f"Your signature on a contract in which they pay you {amt} cents", f"A payment of {amt} cents from you",
                "An answer to a mail", "Your vote on a proposal"]
        return p, q, opts, 0
    if t == "invite_waiting":
        org = f"{_choice(rng, NAMES)}-{_choice(rng, ['studio', 'labs', 'coop', 'works'])}{int(rng.integers(10, 99))}"
        oid = bench.must(other, "id.create_org", {"handle": org, "name": org.title()})["org"]
        since = bench.world.seq
        bench.must(other, "id.invite", {"org": oid, "person": me})
        p = bench.view(me, goal, since)
        q = "What is pending for you?"
        opts = [f"An invitation to join {oid}", f"A contract from {other}", "An unpaid invoice", "An unread mail"]
        return p, q, opts, 0
    if t == "money_received":
        amt = int(rng.integers(5, 95)) * 100
        bench.must(other, "bank.transfer", {"from_account": oacct, "to": me, "amount": amt, "memo": "thanks for the help"})
        p = bench.view(me, goal, since)
        q = "What happened to your money?"
        opts = [f"{other} sent you {amt} cents", f"You sent {other} {amt} cents", f"{other} sent you an invoice over {amt} cents",
                "Nothing changed"]
        return p, q, opts, 0
    raise KeyError(t)


SITUATION_TEMPLATES = {"train": ["afford_invoice", "who_waits", "cheapest_provider", "invite_waiting"],
                       "test_new": ["why_failed", "forum_reply", "contract_waiting", "money_received"]}


def build_situation(bench: Bench, split: str, n: int, rng) -> list[dict]:
    out = []
    for k in range(n):
        held = split == "test" and k % 2 == 0
        t = _choice(rng, SITUATION_TEMPLATES["test_new"] if held else SITUATION_TEMPLATES["train"])
        p, q, opts, ans = _stage_situation(bench, t, rng)
        out.append(_mc("situation_interpretation", split, t, k, held, p, q, opts, ans, rng))
    return out


# ====================================================================== 3. tool choice
# (template, need text, correct tool, staging) - the need never names the tool.
def _need(bench: Bench, t: str, me: str, acct: str, other: str, oacct: str, rng) -> tuple[str, str, list]:
    X = other
    thing = _choice(rng, THINGS)
    prod = _choice(rng, PRODUCTS)
    if t == "pay_debt":
        return f"You owe {X} {int(rng.integers(5, 90)) * 100} cents for the {thing}. Settle the debt now.", "bank.transfer", []
    if t == "check_money":
        return "Before you buy anything, find out how much money is in your accounts.", "bank.accounts", []
    if t == "ask_privately":
        return f"Ask {X} privately whether they can review your {thing} by {_choice(rng, DAYS)}.", "mail.send", []
    if t == "read_new_mail":
        bench.must(other, "mail.send", {"to": [me], "subject": f"About the {thing}", "body": "Please call me."})
        return f"{X} wrote to you about the {thing}. Find out what the message says.", "mail.read", []
    if t == "publish_guide":
        return f"Write down instructions for using your {prod} so that anyone in the world can read them.", "docs.create", []
    if t == "ask_community":
        return f"Ask the community for advice about {_choice(rng, TOPICS)}.", "forum.post", []
    if t == "web_search":
        return f"Find out whether anybody in this world already sells a {prod}.", "search.query", []
    if t == "compare_compute":
        return "Compare what compute providers charge before deciding where to host your service.", "cloud.providers", []
    if t == "new_project":
        return f"Start a new software project called {prod.replace(' ', '-')} to hold your code.", "code.create_repo", []
    if t == "check_tests":
        return f"Check whether the automated tests of your repository {me.split(':')[1]}/{prod.replace(' ', '-')} pass.", "code.test", []
    if t == "schedule":
        return f"Set up a one-hour meeting with {X} next {_choice(rng, DAYS)} at {int(rng.integers(9, 17))}:00.", "cal.create", []
    if t == "find_work":
        return "You need a job. See which positions are currently open.", "jobs.openings", []
    if t == "pay_invoice":
        bench.must(other, "bank.invoice", {"amount": int(rng.integers(5, 50)) * 100, "payer": me, "to_account": oacct,
                                           "memo": thing})
        return f"Pay the invoice {X} sent you for the {thing}.", "bank.pay_invoice", []
    if t == "found_org":
        return f"Found an organisation called {_choice(rng, COMPANIES)} for your team.", "id.create_org", []
    if t == "sign_contract":
        bench.must(other, "contracts.propose", {"parties": [other, me], "title": f"{thing} work",
                                                "clauses": [{"type": "payment", "from": other, "to": me,
                                                             "amount": int(rng.integers(10, 90)) * 100}]})
        return f"{X} sent you a contract for the {thing}. You agree with its terms; accept it.", "contracts.sign", []
    if t == "answer_forum":
        bench.must(other, "forum.post", {"board": "help", "title": f"How do I set up a {prod}?", "body": "Any tips?"})
        return f"{X} asked on the forum how to set up a {prod}, which you know well. Answer them publicly.", "forum.reply", []
    if t == "join_org":
        org = f"{_choice(rng, NAMES)}-team{int(rng.integers(10, 999))}"
        oid = bench.must(other, "id.create_org", {"handle": org, "name": org})["org"]
        bench.must(other, "id.invite", {"org": oid, "person": me})
        return f"You were invited to join {oid}. Join it.", "id.respond_invite", []
    if t == "rent_server":
        return f"Get a server with {int(rng.integers(1, 5))} cores to run your service.", "cloud.rent", []
    if t == "advertise":
        return f"Make more people find your {prod} when they look for one in search results.", "ads.create_campaign", []
    if t == "report_failure":
        return f"The service {_choice(rng, ['route', 'mailer', 'resizer'])}-svc.crucible you use keeps failing. Tell its operator.", "support.open", []
    if t == "file_bug":
        return f"Record a bug in your project tracker {me.split(':')[1]}-tracker: the login page crashes.", "issues.create", []
    if t == "save_code":
        return f"Save your new version of main.py to your repository {me.split(':')[1]}/{prod.replace(' ', '-')}.", "code.commit", []
    if t == "read_doc":
        d = bench.must(other, "docs.create", {"title": f"Notes on the {thing}", "body": "Step one: plan.", "visibility": "public"})["doc"]
        bench.must(other, "mail.send", {"to": [me], "subject": "notes", "body": f"See document {d} for the plan."})
        return f"{X} pointed you to a document with the plan for the {thing}. Read the document.", "docs.get", []
    if t == "hire":
        return f"Your company needs a {_choice(rng, ['developer', 'support agent', 'analyst'])}. Advertise the position publicly.", "jobs.post", []
    if t == "shop":
        return f"You want to buy a ready-made {prod}. Look at what is offered for sale.", "market.browse", []
    if t == "discover":
        return "Vote in favour of the open proposal of your organisation.", "gov.vote", ["gov."]
    raise KeyError(t)


TOOL_TEMPLATES = {"train": ["pay_debt", "check_money", "ask_privately", "read_new_mail", "publish_guide",
                            "ask_community", "web_search", "compare_compute", "new_project", "check_tests",
                            "schedule", "find_work", "pay_invoice", "found_org"],
                  "test_new": ["sign_contract", "answer_forum", "join_org", "rent_server", "advertise",
                               "report_failure", "file_bug", "save_code", "read_doc", "hire", "shop", "discover"]}
N_TOOL_OPTIONS = 10


def tool_candidates(tools: list[dict], situation: str, correct: str, rng, exclude_prefixes=(), k: int = N_TOOL_OPTIONS):
    """Options for one situation: the correct tool (unless excluded: then the right move is discovery), the tools
    most similar to the situation text (hard distractors), random others, plus world.find_tools and wait."""
    from ..participants import WAIT
    pool = [t for t in tools if t["name"] not in ("world.find_tools",) and
            not any(t["name"].startswith(x) for x in exclude_prefixes)]
    by = {t["name"]: t for t in pool}
    names = sorted(by)
    sc = BM25([tool_text(by[n]) for n in names]).scores(situation)
    near = [n for _, n in sorted(zip(sc, names), key=lambda x: (-x[0], x[1])) if n != correct][:5]
    rest = [n for n in names if n != correct and n not in near]
    head = ([correct] if correct in by else []) + near
    pick = head + _sample(rng, rest, max(0, k - 2 - len(head)))
    find = [t for t in tools if t["name"] == "world.find_tools"]
    return [by[n] for n in pick] + find + [WAIT]


def build_tool_choice(bench: Bench, split: str, n: int, rng) -> list[dict]:
    out = []
    for k in range(n):
        held = split == "test"
        t = _choice(rng, TOOL_TEMPLATES["test_new"] if held else TOOL_TEMPLATES["train"])
        me, acct = bench.person("r")
        other, oacct = bench.person(_choice(rng, NAMES))
        since = bench.world.seq
        need, correct, exclude = _need(bench, t, me, acct, other, oacct, rng)
        p = bench.view(me, need, since)
        tools = bench.tools(me)
        cands = tool_candidates(tools, need, correct, rng, exclude)
        order = [int(i) for i in rng.permutation(len(cands) - 2)] + [len(cands) - 2, len(cands) - 1]
        cands = [cands[i] for i in order]
        names = [c["name"] for c in cands]
        target = correct if correct in names else "world.find_tools"
        req = p.choice_request(cands)
        lex = BM25([tool_text(c) for c in cands[:-1]]).scores(need)
        out.append({"id": f"tool_choice/{split}/{t}/{k}", "cap": "tool_choice", "split": split, "template": t,
                    "held_out_template": held, "format": "mc", "messages": req["messages"], "prefix_messages": 1,
                    "query": p.header(), "options": names, "answer": names.index(target),
                    "chance": 1.0 / len(names), "lexical_choice": int(np.argmax(lex)),
                    "option_texts": [tool_text(c) for c in cands],
                    "option_lines": [tool_line(c, p.tool_chars) for c in cands]})
    return out


# ====================================================================== 4. communication
def _comm_facts(rng):
    X = _choice(rng, NAMES)
    day = _choice(rng, DAYS[:5])
    hour = int(rng.integers(9, 18))
    amt = int(rng.integers(10, 99)) * 100
    thing = _choice(rng, THINGS)
    room = _choice(rng, ROOMS)
    return X, day, hour, amt, thing, room


def build_communication(bench: Bench, split: str, n: int, rng) -> list[dict]:
    """Generative: write a mail that conveys given facts (checked automatically); plus a multiple-choice variant
    (which draft states the facts correctly) that every model, including encoders, can answer."""
    out = []
    for k in range(n):
        X, day, hour, amt, thing, room = _comm_facts(rng)
        me, _ = bench.person("r")
        kind = k % 2
        if kind == 0:
            facts = [day, f"{hour}:00", room]
            task = (f"Write a short mail to {X} telling them that the review of the {thing} is on {day} at {hour}:00 "
                    f"in {room}, and asking them to confirm.")
        else:
            facts = [str(amt), thing, day]
            task = (f"Write a short mail to {X} telling them that you paid {amt} cents for the {thing} and that you "
                    f"need the finished work by {day}.")
        ctx = _situation(me, task)
        gen = {"id": f"communication/{split}/write{kind}/{k}", "cap": "communication", "split": split,
               "template": f"write{kind}", "held_out_template": False, "format": "gen",
               "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": ctx + "\n\nReply with the body of the mail only (2-4 sentences)."}],
               "prefix_messages": 1, "max_new_tokens": 120, "eval": {"kind": "facts", "facts": facts, "recipient": X}}
        out.append(gen)
        # discriminative variant
        if kind == 0:
            right = f"Hi {X}, the review of the {thing} is on {day} at {hour}:00 in {room}. Can you confirm?"
            d2 = _choice(rng, [d for d in DAYS[:5] if d != day])
            wrong = [f"Hi {X}, the review of the {thing} is on {d2} at {hour}:00 in {room}. Can you confirm?",
                     f"Hi {X}, the review of the {thing} is on {day} at {hour + 1}:00 in {room}. Can you confirm?",
                     f"Hi {X}, the review of the {thing} is on {day} at {hour}:00 in {_choice(rng, [r for r in ROOMS if r != room])}. Can you confirm?"]
        else:
            right = f"Hi {X}, I paid {amt} cents for the {thing}; I need the finished work by {day}."
            wrong = [f"Hi {X}, I paid {amt + 500} cents for the {thing}; I need the finished work by {day}.",
                     f"Hi {X}, you owe me {amt} cents for the {thing}; I need the finished work by {day}.",
                     f"Hi {X}, I paid {amt} cents for the {thing}; I need the finished work by {_choice(rng, [d for d in DAYS[:5] if d != day])}."]
        q = "Which draft states the facts of your task correctly?"
        out.append(_mc("communication", split, f"choose{kind}", k, False, ctx, q, [right] + wrong, 0, rng))
    return out


# ====================================================================== 5. adaptation to changed incentives
def _incentive(rng, t: str, after: bool):
    if t == "provider_price":
        a, b = _sample(rng, ["o:cloud1", "o:cloud2", "o:nimbus", "o:stratos"], 2)
        pa, pb = int(rng.integers(100, 160)), int(rng.integers(170, 230))
        text = f"You rent servers from {a} at {pa} cents per core-hour. {b} charges {pb} cents per core-hour."
        if after:
            pa2 = pb + int(rng.integers(20, 80))
            text += f" Today {a} announced that its price rises to {pa2} cents per core-hour for all new rentals."
        q = "Where should you rent your next server to pay the least?"
        return text, q, [a, b, "It makes no difference", "Do not rent any server"], (1 if after else 0)
    if t == "bounty_reward":
        x, y = int(rng.integers(30, 60)) * 100, int(rng.integers(61, 90)) * 100
        text = (f"Two bounties take you the same time: translating a guide pays {x} cents, reviewing code pays {y} "
                f"cents. You can do only one this week.")
        if after:
            text += f" The guide's author has just raised the translation reward to {y + int(rng.integers(10, 40)) * 100} cents."
        q = "Which bounty should you take to earn the most?"
        return text, q, ["the code review", "the translation", "both", "neither"], (1 if after else 0)
    if t == "late_penalty":
        c = int(rng.integers(5, 20)) * 100
        f = int(rng.integers(30, 60)) * 100
        d = int(rng.integers(2, 5))
        text = (f"You can deliver an order today if you pay {c} cents for express shipping, or deliver it {d} days late "
                f"for free.")
        if after:
            text += f" The customer's new contract charges you {f} cents for every day of delay."
        else:
            text += " The customer's contract has no penalty for delays."
        q = "What is the cheaper option for you?"
        return text, q, ["deliver late for free", "pay for express shipping", "cancel the order", "they cost the same"], (1 if after else 0)
    if t == "market_fee":
        p = int(rng.integers(50, 200)) * 100
        f1, f2 = int(rng.integers(2, 6)), int(rng.integers(8, 14))
        text = f"You sell a {_choice(rng, PRODUCTS)} for {p} cents. Marketplace A takes a {f1} percent fee, marketplace B takes {f2} percent."
        if after:
            text += f" Marketplace A now also charges a fixed {int(p * (f2 - f1 + 3) / 100)} cents per sale."
        q = "Where do you keep more money from each sale?"
        return text, q, ["marketplace A", "marketplace B", "the same on both", "neither allows sales"], (1 if after else 0)
    if t == "subsidy":
        g, e = int(rng.integers(300, 500)) * 100, int(rng.integers(520, 700)) * 100
        text = f"A petrol delivery van costs {g} cents and an electric one {e} cents; running costs are the same."
        if after:
            text += f" The city now pays a subsidy of {e - g + int(rng.integers(20, 80)) * 100} cents for every electric van bought."
        q = "Which van is cheaper for you to buy?"
        return text, q, ["the petrol van", "the electric van", "they cost the same", "neither is for sale"], (1 if after else 0)
    if t == "ad_price":
        cpa, cpb = int(rng.integers(5, 15)), int(rng.integers(20, 40))
        text = (f"Search ads on keyword 'hosting' cost {cpa} cents per impression and bring as many customers as ads on "
                f"'servers', which cost {cpb} cents.")
        if after:
            text += f" Competition just pushed the price of 'hosting' ads to {cpb + int(rng.integers(10, 30))} cents."
        q = "Which keyword should you advertise on now?"
        return text, q, ["hosting", "servers", "both cost the same", "advertising is not possible"], (1 if after else 0)
    raise KeyError(t)


INCENTIVE_TEMPLATES = {"train": ["provider_price", "bounty_reward", "late_penalty"],
                       "test_new": ["market_fee", "subsidy", "ad_price"]}


def build_incentive(bench: Bench, split: str, n: int, rng) -> list[dict]:
    """Pairs of items: the same situation before and after an incentive changes (the right choice flips)."""
    out = []
    for k in range(0, n, 2):
        held = split == "test" and (k // 2) % 2 == 0
        t = _choice(rng, INCENTIVE_TEMPLATES["test_new"] if held else INCENTIVE_TEMPLATES["train"])
        me = f"p:r{k}x"
        st = int(rng.integers(0, 2 ** 31))
        for after in (False, True):
            r2 = np.random.default_rng(st)
            text, q, opts, ans = _incentive(r2, t, after)
            it = _mc("incentive_adaptation", split, t, k + int(after), held, _situation(me, text), q, opts, ans, rng,
                     extra={"pair": f"{split}/{k}", "after": after})
            out.append(it)
    return out


# ====================================================================== 6. negotiation
def _negotiation(rng, t: str):
    if t == "sell_service":
        m = int(rng.integers(20, 80)) * 100
        if rng.random() < 0.5:
            o = m + int(rng.integers(1, 30)) * 100
        else:
            o = m - int(rng.integers(5, 15)) * 100
        hi = m + int(rng.integers(10, 40)) * 100
        text = (f"You sell a {_choice(rng, PRODUCTS)} set-up service. Doing the job costs you {m} cents, so you must not "
                f"accept less. A customer offers {o} cents.")
        role = "seller"
    elif t == "buy_licence":
        m = int(rng.integers(40, 120)) * 100     # your maximum budget
        o = m - int(rng.integers(1, 20)) * 100 if rng.random() < 0.5 else m + int(rng.integers(5, 20)) * 100
        text = (f"You want to buy a software licence and can pay at most {m} cents. The seller asks {o} cents.")
        role = "buyer"
        hi = m
    elif t == "salary":
        m = int(rng.integers(30, 80)) * 10000
        o = m + int(rng.integers(1, 10)) * 10000 if rng.random() < 0.5 else m - int(rng.integers(2, 10)) * 10000
        hi = m + 20 * 10000
        text = (f"You are offered a job. You will not work for less than {m} cents per month (your other offer pays "
                f"that). The employer offers {o} cents per month.")
        role = "seller"
    elif t == "split_profit":
        tot = int(rng.integers(100, 300)) * 100
        m = int(tot * rng.uniform(0.3, 0.45)) // 100 * 100   # your outside option
        o = (m + int(rng.integers(1, 20)) * 100) if rng.random() < 0.5 else max(100, m - int(rng.integers(5, 20)) * 100)
        hi = tot
        text = (f"You and a partner earned {tot} cents together. Working alone you would have earned {m} cents. Your "
                f"partner proposes that you get {o} cents.")
        role = "seller"
    else:
        raise KeyError(t)
    if role == "seller":
        if o >= m:
            opts = ["Accept the offer", f"Counter-offer {max(100, o - int(rng.integers(5, 15)) * 100)} cents",
                    "Reject and walk away", f"Counter-offer {max(100, m - int(rng.integers(5, 15)) * 100)} cents"]
        else:
            c = min(hi, m + int(rng.integers(1, 10)) * 100)
            opts = [f"Counter-offer {c} cents", "Accept the offer",
                    f"Counter-offer {max(100, (o + m) // 2 - (m - o))} cents", "Reject and walk away"]
    else:
        if o <= m:
            opts = ["Accept the price", f"Counter-offer {o + int(rng.integers(5, 15)) * 100} cents",
                    "Reject and walk away", f"Counter-offer {m + int(rng.integers(5, 15)) * 100} cents"]
        else:
            c = m - int(rng.integers(1, 10)) * 100
            opts = [f"Counter-offer {c} cents", "Accept the price", f"Counter-offer {o + int(rng.integers(5, 15)) * 100} cents",
                    "Reject and walk away"]
    return text, opts, {"role": role, "limit": m, "offer": o, "cap": hi}


NEGOTIATION_TEMPLATES = {"train": ["sell_service", "buy_licence"], "test_new": ["salary", "split_profit"]}


def build_negotiation(bench: Bench, split: str, n: int, rng) -> list[dict]:
    """Multiple choice (exactly one move is rational and keeps a deal possible) and a generative variant: reply
    with JSON {"action": accept|counter|reject, "amount": cents}, checked for rationality."""
    out = []
    for k in range(n):
        held = split == "test" and k % 2 == 0
        t = _choice(rng, NEGOTIATION_TEMPLATES["test_new"] if held else NEGOTIATION_TEMPLATES["train"])
        text, opts, meta = _negotiation(rng, t)
        me = f"p:r{k}x"
        ctx = _situation(me, text)
        q = "What is your best move? You want a deal, but never one that leaves you worse off than your limit."
        out.append(_mc("negotiation", split, t, k, held, ctx, q, opts, 0, rng))
        out.append({"id": f"negotiation/{split}/{t}/gen{k}", "cap": "negotiation", "split": split, "template": t,
                    "held_out_template": held, "format": "gen",
                    "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                 {"role": "user", "content": ctx + "\n\n" + q + ' Reply with JSON only: {"action": '
                                  '"accept" or "counter" or "reject", "amount": integer cents or null}.'}],
                    "prefix_messages": 1, "max_new_tokens": 40, "eval": {"kind": "negotiation", **meta}})
    return out


# ====================================================================== 7. programming
PROGRAMS = [  # (name, signature, spec, reference, tests [(args, expected)])
    ("word_count", "word_count(text)", "return the number of whitespace-separated words in text",
     "def word_count(text):\n    return len(text.split())\n", [(["a b c"], 3), ([""], 0), (["  hello   world "], 2)]),
    ("reverse_words", "reverse_words(text)", "return the words of text in reverse order, joined by single spaces",
     "def reverse_words(text):\n    return ' '.join(reversed(text.split()))\n",
     [(["a b c"], "c b a"), (["one"], "one"), (["  x  y "], "y x")]),
    ("is_palindrome", "is_palindrome(s)", "return True if s reads the same backwards, ignoring case and any "
     "character that is not a letter or digit", "def is_palindrome(s):\n    t = [c.lower() for c in s if c.isalnum()]\n    return t == t[::-1]\n",
     [(["Anna"], True), (["A man, a plan, a canal: Panama"], True), (["abc"], False)]),
    ("fizzbuzz", "fizzbuzz(n)", "return a list of strings for 1..n: 'Fizz' for multiples of 3, 'Buzz' for multiples "
     "of 5, 'FizzBuzz' for multiples of both, otherwise the number as a string",
     "def fizzbuzz(n):\n    out = []\n    for i in range(1, n + 1):\n        s = ('Fizz' if i % 3 == 0 else '') + ('Buzz' if i % 5 == 0 else '')\n        out.append(s or str(i))\n    return out\n",
     [([5], ["1", "2", "Fizz", "4", "Buzz"]), ([15], ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz"])]),
    ("total_cents", "total_cents(items)", "items is a list of dicts with keys 'price_cents' and 'qty'; return the "
     "total price in cents", "def total_cents(items):\n    return sum(i['price_cents'] * i['qty'] for i in items)\n",
     [([[{"price_cents": 250, "qty": 2}, {"price_cents": 100, "qty": 3}]], 800), ([[]], 0)]),
    ("apply_discount", "apply_discount(price_cents, percent)", "return the price after a discount of percent "
     "percent, rounded down to whole cents", "def apply_discount(price_cents, percent):\n    return price_cents * (100 - percent) // 100\n",
     [([1000, 15], 850), ([999, 10], 899), ([500, 0], 500)]),
    ("count_vowels", "count_vowels(text)", "return how many vowels (a, e, i, o, u, upper or lower case) text contains",
     "def count_vowels(text):\n    return sum(1 for c in text.lower() if c in 'aeiou')\n", [(["Hello World"], 3), (["xyz"], 0), (["AEIOU"], 5)]),
    ("max_profit", "max_profit(prices)", "prices is a list of daily prices; return the largest prices[j] - prices[i] "
     "with j > i, or 0 if no profit is possible",
     "def max_profit(prices):\n    best, low = 0, None\n    for p in prices:\n        if low is not None:\n            best = max(best, p - low)\n        low = p if low is None else min(low, p)\n    return best\n",
     [([[7, 1, 5, 3, 6, 4]], 5), ([[7, 6, 4, 3, 1]], 0), ([[]], 0)]),
    ("dedupe", "dedupe(items)", "return the list without duplicates, keeping the first occurrence of each item in order",
     "def dedupe(items):\n    seen, out = set(), []\n    for x in items:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out\n",
     [([[3, 1, 3, 2, 1]], [3, 1, 2]), ([[]], [])]),
    ("parse_kv", "parse_kv(text)", "text looks like 'a=1;b=2'; return a dict mapping each key to its value as a "
     "string (empty text gives an empty dict)",
     "def parse_kv(text):\n    out = {}\n    for part in text.split(';'):\n        if part:\n            k, v = part.split('=', 1)\n            out[k] = v\n    return out\n",
     [(["a=1;b=2"], {"a": "1", "b": "2"}), ([""], {}), (["x=y"], {"x": "y"})]),
    ("median", "median(nums)", "return the median of a non-empty list of numbers (the mean of the two middle values "
     "for an even count)",
     "def median(nums):\n    s = sorted(nums)\n    n = len(s)\n    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2\n",
     [([[3, 1, 2]], 2), ([[4, 1, 2, 3]], 2.5), ([[5]], 5)]),
    ("running_balance", "running_balance(start, moves)", "return the list of balances after applying each amount in "
     "moves (positive or negative) to start, in order",
     "def running_balance(start, moves):\n    out = []\n    for m in moves:\n        start += m\n        out.append(start)\n    return out\n",
     [([100, [50, -30, 10]], [150, 120, 130]), ([0, []], [])]),
    ("chunk", "chunk(items, size)", "split items into consecutive lists of length size (the last may be shorter)",
     "def chunk(items, size):\n    return [items[i:i + size] for i in range(0, len(items), size)]\n",
     [([[1, 2, 3, 4, 5], 2], [[1, 2], [3, 4], [5]]), ([[], 3], [])]),
    ("caesar", "caesar(text, shift)", "shift every lowercase letter a-z forward by shift places in the alphabet, "
     "wrapping around after z; leave every other character unchanged",
     "def caesar(text, shift):\n    return ''.join(chr((ord(c) - 97 + shift) % 26 + 97) if 'a' <= c <= 'z' else c for c in text)\n",
     [(["abc", 1], "bcd"), (["xyz", 3], "abc"), (["a-b!", 2], "c-d!")]),
    ("word_freq", "word_freq(text)", "return a dict mapping each lowercase word of text to how often it occurs",
     "def word_freq(text):\n    out = {}\n    for w in text.lower().split():\n        out[w] = out.get(w, 0) + 1\n    return out\n",
     [(["a B a"], {"a": 2, "b": 1}), ([""], {})]),
    ("second_largest", "second_largest(nums)", "return the second largest distinct value of nums, or None if there "
     "is none", "def second_largest(nums):\n    s = sorted(set(nums))\n    return s[-2] if len(s) >= 2 else None\n",
     [([[3, 1, 3, 2]], 2), ([[5, 5]], None), ([[1, 9]], 1)]),
    ("to_snake", "to_snake(name)", "convert a CamelCase name to snake_case (lowercase, words joined by underscores)",
     "def to_snake(name):\n    out = ''\n    for i, c in enumerate(name):\n        if c.isupper() and i > 0:\n            out += '_'\n        out += c.lower()\n    return out\n",
     [(["HelloWorld"], "hello_world"), (["A"], "a"), (["OrderID"], "order_i_d")]),
    ("clamp", "clamp(x, lo, hi)", "return x limited to the range lo..hi", "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n",
     [([5, 0, 10], 5), ([-3, 0, 10], 0), ([42, 0, 10], 10)]),
    ("average_rating", "average_rating(ratings)", "return the mean of a list of ratings rounded to one decimal, or 0.0 "
     "for an empty list", "def average_rating(ratings):\n    return round(sum(ratings) / len(ratings), 1) if ratings else 0.0\n",
     [([[4, 5, 3]], 4.0), ([[]], 0.0), ([[1, 2]], 1.5)]),
    ("count_overdue", "count_overdue(invoices, today)", "invoices is a list of dicts with keys 'due' (a day number) "
     "and 'paid' (bool); return how many are unpaid and due before today",
     "def count_overdue(invoices, today):\n    return sum(1 for i in invoices if not i['paid'] and i['due'] < today)\n",
     [([[{"due": 3, "paid": False}, {"due": 9, "paid": False}, {"due": 1, "paid": True}], 5], 1), ([[], 5], 0)]),
    ("cheapest", "cheapest(offers)", "offers is a dict mapping seller name to price; return the name of the cheapest "
     "seller (the alphabetically first one if several are equally cheap)",
     "def cheapest(offers):\n    return min(sorted(offers), key=lambda k: offers[k])\n",
     [([{"b": 5, "a": 7}], "b"), ([{"x": 3, "c": 3}], "c")]),
    ("initials", "initials(name)", "return the uppercase initials of the words of a full name, without separators",
     "def initials(name):\n    return ''.join(w[0].upper() for w in name.split())\n",
     [(["ada lovelace"], "AL"), (["Grace Brewster Hopper"], "GBH"), ([""], "")]),
    ("rotate", "rotate(items, k)", "rotate the list right by k positions (k may be larger than the length)",
     "def rotate(items, k):\n    if not items:\n        return []\n    k %= len(items)\n    return items[-k:] + items[:-k] if k else list(items)\n",
     [([[1, 2, 3, 4], 1], [4, 1, 2, 3]), ([[1, 2, 3], 5], [2, 3, 1]), ([[], 2], [])]),
    ("valid_handle", "valid_handle(h)", "return True if h starts with a lowercase letter and has 2 to 31 characters, "
     "all lowercase letters, digits or '-'",
     "import re\n\ndef valid_handle(h):\n    return re.fullmatch(r'[a-z][a-z0-9-]{1,30}', h) is not None\n",
     [(["ana"], True), (["A1"], False), (["x"], False), (["b-2"], True)]),
]
PROGRAM_SPLIT = {"train": list(range(0, 8)), "dev": [8, 9], "test": list(range(10, len(PROGRAMS)))}


def _mutants(src: str) -> list[str]:
    """Buggy variants of a reference solution (for repair tasks and the multiple-choice variant)."""
    subs = [(" + 1", " + 2"), ("- 1", "+ 1"), ("<", "<="), (">", ">="), ("min(", "max("), ("max(", "min("),
            ("//", "/"), (" % 2", " % 3"), ("[::-1]", ""), ("reversed(", "sorted("), (".lower()", ""),
            ("sum(", "len("), ("[-2]", "[-1]"), ("out.append(start)", "out.append(m)"), ("100 - percent", "percent"),
            ("not i['paid'] and ", ""), ("'aeiou'", "'aeio'"), (" * i['qty']", ""), ("items[-k:] + items[:-k]", "items[k:] + items[:k]"),
            ("i > 0", "i >= 0"), ("w[0].upper()", "w[0]"), ("{1,30}", "{1,31}"), ("(ord(c) - 97 + shift) % 26", "(ord(c) - 97 + shift)"),
            ("seen.add(x)", "pass"), ("if part:", "if True:"), ("k, v = part.split('=', 1)", "v, k = part.split('=', 1)")]
    subs += [("True", "False"), ("False", "True"), ("==", "!="), ("+=", "-="), ("return ", "return -"),
             ("[0]", "[1]"), ("range(1, n + 1)", "range(1, n)"), ("i % 3", "i % 4"), ("'Buzz'", "'Fizz'"),
             ("split()", "split(' ')"), ("sorted(", "list("), ("a=", "b="), ("if x not in seen", "if x in seen")]
    out = []
    for a, b in subs:
        if a in src:
            m = src.replace(a, b, 1)
            if m != src and m not in out:
                out.append(m)
    for mt in re.finditer(r"(?<![\w.])(\d+)(?![\w.])", src):   # off-by-one on integer literals
        m = src[:mt.start()] + str(int(mt.group(1)) + 1) + src[mt.end():]
        if m not in out:
            out.append(m)
    return out


def _run_cases(files: dict, name: str, cases) -> dict:
    """Execute `name` on the cases in the Crucible sandbox; returns {passed, total, first_failure}."""
    from ...sandbox.runner import run_program
    harness = (f"import solution\n\ndef run(args, caller):\n    out = []\n    for a in args:\n        try:\n"
               f"            out.append({{'ok': True, 'v': solution.{name}(*a)}})\n        except Exception as e:\n"
               f"            out.append({{'ok': False, 'v': type(e).__name__ + ': ' + str(e)[:120]}})\n    return out\n")
    res = run_program(dict(files, **{"_bench_harness.py": harness}), "_bench_harness:run", [list(a) for a, _ in cases],
                      timeout=10)
    if not res.get("ok"):
        err = res.get("error") or {}
        return {"passed": 0, "total": len(cases), "first_failure": f"the module failed to load: {err.get('type', '')}: "
                f"{str(err.get('message', ''))[:160]}"}
    got = res.get("result") or []
    passed, first = 0, None
    for (a, want), g in zip(cases, got):
        if g.get("ok") and g.get("v") == want:
            passed += 1
        elif first is None:
            shown = ", ".join(json.dumps(x) for x in a)
            first = (f"{name}({shown}) raised {g['v']}" if not g.get("ok") else
                     f"{name}({shown}) returned {json.dumps(g.get('v'))}, expected {json.dumps(want)}")
    return {"passed": passed, "total": len(cases), "first_failure": first}


def build_programming(bench: Bench, split: str, n: int, rng) -> list[dict]:
    out = []
    ids = PROGRAM_SPLIT[split]
    for j, pi in enumerate(ids):
        name, sig, spec, ref, cases = PROGRAMS[pi]
        me = f"p:dev{pi}x"
        task = (f"Write a Python module solution.py with a function {sig} that should {spec}. The module is run in a "
                f"sandbox that has only the Python standard library.")
        out.append({"id": f"programming/{split}/write/{name}", "cap": "programming", "split": split, "template": "write",
                    "held_out_template": split == "test", "format": "gen",
                    "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                 {"role": "user", "content": _situation(me, task) + "\n\nReply with the complete "
                                  "contents of solution.py in one ```python code block and nothing else."}],
                    "prefix_messages": 1, "max_new_tokens": 320, "eval": {"kind": "program", "name": name, "cases": cases}})
        muts = _mutants(ref)
        good = []
        for m in muts:   # keep mutants that really fail a test (executed in the sandbox)
            r = _run_cases({"solution.py": m}, name, cases)
            if r["passed"] < r["total"]:
                good.append((m, r["first_failure"]))
        if not good:
            continue
        bug, fail = good[int(rng.integers(0, len(good)))]
        task = (f"The function {sig} in solution.py should {spec}. A test fails: {fail}. Here is the current "
                f"solution.py:\n```python\n{bug}```\nFix the bug.")
        out.append({"id": f"programming/{split}/repair/{name}", "cap": "programming", "split": split,
                    "template": "repair", "held_out_template": split == "test", "format": "gen",
                    "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                 {"role": "user", "content": _situation(me, task) + "\n\nReply with the complete "
                                  "corrected solution.py in one ```python code block and nothing else."}],
                    "prefix_messages": 1, "max_new_tokens": 320, "eval": {"kind": "program", "name": name, "cases": cases}})
        if len(good) >= 3:
            opts = [ref] + [g[0] for g in good[:3]]
            q = f"Which version of solution.py correctly implements {sig} so that it {spec}?"
            ctx = _situation(me, "You review code before it is deployed.")
            out.append(_mc("programming", split, "choose", j, split == "test", ctx, q, opts, 0, rng))
    return out


# ====================================================================== 8. strategic decisions
def _strategy(rng, t: str):
    if t == "pricing":
        c = int(rng.integers(2, 6)) * 100
        prices = sorted(_sample(rng, [p * 100 for p in range(6, 21)], 4))
        q0 = int(rng.integers(300, 600))
        slope = rng.uniform(15, 30)
        demand = [max(0, int(q0 - (p / 100) * slope)) for p in prices]
        prof = [(p - c) * d for p, d in zip(prices, demand)]
        best = int(np.argmax(prof))
        text = (f"Each unit of your {_choice(rng, PRODUCTS)} costs you {c} cents to serve. A market survey estimates "
                f"monthly sales at each price: " + "; ".join(f"{p} cents -> {d} units" for p, d in zip(prices, demand)) + ".")
        return text, "Which price gives the highest monthly profit?", [f"{p} cents" for p in prices], best
    if t == "capacity":
        d = int(rng.integers(20, 60))
        m = int(rng.integers(150, 400))
        cost = int(rng.integers(40, 120))
        opts_n = sorted(set([max(1, d - 10), d, d + 15, d + 40]))
        while len(opts_n) < 4:
            opts_n.append(opts_n[-1] + 7)
        prof = [min(n, d) * m - n * cost for n in opts_n]
        best = int(np.argmax(prof))
        text = (f"You expect exactly {d} paying customers next month; each brings {m} cents. Each server slot you rent "
                f"costs {cost} cents and serves one customer; unserved customers are lost.")
        return text, "How many slots should you rent?", [f"{n} slots" for n in opts_n], best
    if t == "repeated_game":
        r = int(rng.integers(20, 100))
        text = (f"You and a rival firm set prices every week for the next {r} weeks. If both keep prices high, each "
                f"earns 300 cents a week; if one undercuts, it earns 500 and the other 0 that week; if both undercut, "
                f"each earns 100. The rival keeps prices high as long as you did the week before and undercuts right "
                f"after you undercut.")
        return text, "What should you do this week to earn the most over all the weeks?", [
            "keep prices high", "undercut every week", "undercut now and then keep high", "leave the market"], 0
    if t == "auction":
        v = int(rng.integers(30, 90)) * 100
        text = (f"A domain name is sold in a sealed-bid auction where the highest bidder wins but pays the second "
                f"highest bid. The domain is worth exactly {v} cents to you.")
        return text, "What should you bid?", [f"{v} cents", f"{v // 2} cents", f"{v * 2} cents", "nothing"], 0
    if t == "entry":
        fixed = int(rng.integers(50, 150)) * 1000
        units = int(rng.integers(100, 400))
        margin = int(rng.integers(100, 900))
        prof = units * margin - fixed
        text = (f"Entering a new market costs a one-time {fixed} cents. You would sell {units} units at a margin of "
                f"{margin} cents each before the market closes for good.")
        return text, "Should you enter?", ["Yes, the margin covers the entry cost", "No, you would lose money",
                                           "Only if a competitor enters", "Wait one month and decide again"], (0 if prof > 0 else 1)
    if t == "risky_project":
        a_win, a_p = int(rng.integers(40, 90)) * 100, int(rng.integers(3, 9)) / 10
        b_val = int(rng.integers(20, 60)) * 100
        ev = a_win * a_p
        text = (f"Project A pays {a_win} cents with probability {a_p:.1f} and nothing otherwise. Project B pays a sure "
                f"{b_val} cents. You are risk-neutral and can do one.")
        return text, "Which project has the higher expected payoff?", ["Project A", "Project B", "Both are equal",
                                                                         "Neither pays anything"], (0 if ev > b_val else 1) if ev != b_val else 2
    raise KeyError(t)


STRATEGY_TEMPLATES = {"train": ["pricing", "capacity", "auction"], "test_new": ["repeated_game", "entry", "risky_project"]}


def build_strategy(bench: Bench, split: str, n: int, rng) -> list[dict]:
    out = []
    for k in range(n):
        held = split == "test" and k % 2 == 0
        t = _choice(rng, STRATEGY_TEMPLATES["test_new"] if held else STRATEGY_TEMPLATES["train"])
        text, q, opts, ans = _strategy(rng, t)
        out.append(_mc("strategic_decisions", split, t, k, held, _situation(f"p:r{k}x", text), q, opts, ans, rng))
    return out


# ====================================================================== assembly
BUILDERS = {"content_understanding": build_content, "situation_interpretation": build_situation,
            "tool_choice": build_tool_choice, "communication": build_communication,
            "incentive_adaptation": build_incentive, "negotiation": build_negotiation,
            "programming": build_programming, "strategic_decisions": build_strategy}
SIZES = {"train": 240, "dev": 40, "test": 120}
GEN_SIZES = {"train": 0, "dev": 8, "test": 40}


def build(cap: str, split: str, n: int | None = None, seed: int = 0) -> list[dict]:
    """Items of one capability and split (deterministic in `seed`)."""
    salt = {"train": 11, "dev": 23, "test": 37}[split]
    rng = np.random.default_rng([seed, salt, sum(map(ord, cap))])
    bench = Bench(seed * 1000 + salt)
    if n is None:   # communication and negotiation make two items (generative and multiple choice) per instance
        n = SIZES[split] // 2 if cap in ("communication", "negotiation") else SIZES[split]
    return BUILDERS[cap](bench, split, n, rng)


def build_all(seed: int = 0, caps=None, splits=SPLITS, sizes: dict | None = None) -> dict[str, list[dict]]:
    from . import CAPABILITIES
    out: dict[str, list[dict]] = {}
    for cap in caps or CAPABILITIES:
        items = []
        for sp in splits:
            n = (sizes or {}).get(sp)
            items += build(cap, sp, n, seed)
        out[cap] = items
    return out


def check_generation(item: dict, text: str) -> dict:
    """Automatic check of a generated answer: {ok, details}."""
    ev = item["eval"]
    if ev["kind"] == "facts":
        t = text.strip()
        words = re.findall(r"[A-Za-z]+", t)
        toks = t.split()
        low = t.lower()
        facts_ok = all(f.lower() in low for f in ev["facts"])
        grams = [" ".join(toks[i:i + 4]).lower() for i in range(max(0, len(toks) - 3))]
        rep = max((grams.count(g) for g in set(grams)), default=0)
        coherent = 5 <= len(toks) <= 160 and rep <= 2 and (len(words) / max(1, len(toks))) >= 0.6
        return {"ok": bool(facts_ok and coherent), "facts_ok": facts_ok, "coherent": coherent, "words": len(toks)}
    if ev["kind"] == "negotiation":
        from ..participants import parse_json_object
        o = parse_json_object(text) or {}
        act = str(o.get("action", "")).lower()
        amt = o.get("amount")
        m, off, cap = ev["limit"], ev["offer"], ev["cap"]
        if ev["role"] == "seller":
            good = (act == "accept" and off >= m) or (act == "counter" and off < m and isinstance(amt, int) and m <= amt <= cap)
        else:
            good = (act == "accept" and off <= m) or (act == "counter" and off > m and isinstance(amt, int) and 0 < amt <= m)
        return {"ok": bool(good), "action": act, "amount": amt if isinstance(amt, (int, float)) else None}
    if ev["kind"] == "program":
        m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
        code = m[0] if m else text
        r = _run_cases({"solution.py": code}, ev["name"], ev["cases"])
        return {"ok": r["passed"] == r["total"], **r}
    raise KeyError(ev["kind"])
