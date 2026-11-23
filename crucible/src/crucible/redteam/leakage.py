"""Privileged-information probes.

`EndpointScan` is the broad one: an agent with no relationships calls every read-only tool with every identifier
it can harvest or guess, reads everything it is shown, and the evaluator then searches all of that for *canaries*:
exact values of hidden parameters and of other participants' private data that the evaluator collected from
authoritative state beforehand. The other probes are targeted oracles."""
from __future__ import annotations

import json
import re
from typing import Any

from .framework import DAY, HELD, HOUR, LEAK, Attacker, Evaluator, Operator, Probe, Verdict
from .helpers import found_company, npc_firms

ID_RE = re.compile(r"\b(?:[a-z]{2,10}_\d{6,8}|[pos]:[a-z][a-z0-9-]{1,30}|svc:[a-z0-9-]{1,31}|[a-z][a-z0-9-]{1,30}/[a-z][a-z0-9_-]{0,39})\b")
PREFIXES = ("acct", "doc", "inv", "ctr", "claim", "lst", "ord", "grant", "tx", "dep", "inst", "job", "bounty", "flow",
            "trg", "deal", "cap", "inst", "ev", "adc", "tkt", "iss", "proj", "post", "msg", "sg", "mandate", "prop",
            "wfjob", "req", "resp", "cal", "hw")


def distinctive(v: Any) -> str | None:
    """A value is usable as a canary only if a chance match is implausible."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, int):
        return str(v) if abs(v) >= 10_000 and v % 1000 != 0 and v % 100 != 0 else None
    if isinstance(v, float):
        s = repr(v)
        return s if len(s.replace("-", "").replace(".", "").lstrip("0")) >= 9 and "e" not in s else None
    if isinstance(v, str):
        return v if len(v) >= 12 else None
    return None


def _walk(x: Any, path: str, out: dict[str, str]) -> None:
    if isinstance(x, dict) or hasattr(x, "items"):
        for k, v in x.items():
            _walk(v, f"{path}.{k}", out)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(list(x)[:50]):
            _walk(v, f"{path}[{i}]", out)
    else:
        d = distinctive(x)
        if d is not None:
            out.setdefault(d, path)


def hidden_canaries(world, exclude_owners: set[str]) -> dict[str, str]:
    """{needle: what it is}. Hidden parameters of simulated actors, credentials, private balances, hidden
    consumer attributes, the world seed."""
    st = world.state
    out: dict[str, str] = {}
    for table, fields in (("npc_workers", ("reservation", "skills", "blacklist")), ("npc_firms", ("policy",)),
                          ("npc_lenders", ("rate",)), ("credentials", ("salt", "hash")),
                          ("minds", ("traits", "trust", "beliefs", "reservation", "affinity", "greed", "labor_value",
                                     "effort_cost", "memory"))):
        if table not in st.tables:
            continue
        for rid, rec in st.scan(table):
            for f in fields:
                if f in rec:
                    _walk(rec[f], f"hidden {table}[{rid}].{f}", out)
    for aid, acc in st.scan("accounts"):
        if acc["owner"] not in exclude_owners and acc["owner"] != "sys":
            d = distinctive(acc["balance"])
            if d:
                out.setdefault(d, f"private balance of {acc['owner']} ({aid})")
    # other organisations' private operating figures (their own analytics) and the market's internal summaries
    hosts = {h: r["owner"] for h, r in st.scan("hosts")} if "hosts" in st.tables else {}
    for table, owner_of in (("ops_days", lambda k, r: k.split("|")[0]), ("svc_days", lambda k, r: hosts.get(r.get("host"))),
                            ("cons_summary", lambda k, r: None)):
        if table not in st.tables:
            continue
        for rid, rec in st.scan(table):
            if owner_of(rid, rec) in exclude_owners:
                continue
            figs: dict[str, str] = {}
            _walk({k: v for k, v in rec.items() if k not in ("host", "slot", "t")}, f"private {table}[{rid}]", figs)
            for needle, label in figs.items():             # figures only: the names in these records are public
                if needle.replace(".", "").replace("-", "").isdigit():
                    out.setdefault(needle, label)
    for name in sorted(getattr(st, "arrays", {})):
        arr = st.array(name)
        if arr.dtype.kind == "f" and arr.size:
            for v in arr.reshape(-1)[:40].tolist():
                d = distinctive(float(v))
                if d:
                    out.setdefault(d, f"hidden consumer attribute {name}")
    d = distinctive(world.spec.seed)
    if d:
        out.setdefault(d, "world seed")
    out.setdefault(world.spec.hash(), "world spec hash")
    cfg: dict[str, str] = {}
    for m in world.spec.modules:           # hidden generator configuration (firm policies, consumer parameters ...)
        if m["name"] in ("consumers", "civ", "society", "shocks", "rig", "use"):
            c = {k: v for k, v in m["config"].items() if k not in ("bounties", "needs", "providers")}   # public by design
            _walk(c, f"hidden config of module {m['name']}", cfg)
    for needle, label in cfg.items():      # numbers only: names in the configuration (needs, handles) are public
        if needle.replace(".", "").replace("-", "").isdigit():
            out.setdefault(needle, label)
    return out


def find_canaries(text: str, canaries: dict[str, str]) -> list[tuple[str, str]]:
    """Canaries present in `text`. A numeric canary must stand alone as a number: digits inside identifiers, hashes
    or longer numbers are chance matches, not leaks."""
    hits = []
    for needle, label in canaries.items():
        numeric = needle.replace(".", "").replace("-", "").isdigit()
        i = text.find(needle)
        while i >= 0:
            if not numeric:
                hits.append((needle, label))
                break
            before = text[i - 1] if i > 0 else " "
            j = i + len(needle)
            after = text[j] if j < len(text) else " "
            after2 = text[j + 1] if j + 1 < len(text) else " "
            if not (before.isalnum() or before in "._-") and not after.isalnum() and after != "_" \
                    and not (after == "." and after2.isdigit()):
                hits.append((needle, label))
                break
            i = text.find(needle, i + 1)
    return hits


class EndpointScan(Probe):
    name = "leak.scan.read_endpoints_and_observations"
    attack_class = "information_leakage/endpoint_scan"
    world = {"open_world": True, "consumers": True, "seed": 918273, "params": {"max_actions_per_hour": 0}}
    slow = True
    description = ("After one simulated day of an open digital economy (firms run by minds, consumers, lenders, "
                   "workers) in which an independent participant keeps private documents, mail, code, a support "
                   "ticket, a calendar entry and a contract, a newly arrived agent with no relationships calls "
                   "every read-only tool with every identifier it can harvest from earlier answers or guess from "
                   "the sequential id patterns, reads its whole observation stream and every error message. "
                   "Checked: whether any exact value of a hidden parameter (reservation wages, firm policies, mind "
                   "traits and trust, consumer attributes, credentials, generator configuration, the world seed), "
                   "any other principal's balance, or any planted private string appears in what it received.")

    def setup(self, op: Operator) -> None:
        w = op.world
        self.att = op.join("scanner", 300_000)
        v = op.join("victim", 1_234_567, attacker=False, kind="human")
        self.victim = v
        self.planted: dict[str, str] = {}

        def plant(kind: str) -> str:
            s = f"CANARY-{kind}-7f3a91c2e5"
            self.planted[s] = f"private {kind} of {v}"
            return s
        op.as_(v, "docs.create", title="plan", body=plant("document"))
        op.as_(v, "mail.send", to=["victim"], subject="note to self", body=plant("mail"))
        repo = op.as_(v, "code.create_repo", name="secret")["repo"]
        op.as_(v, "code.commit", repo=repo, files={"key.py": f"KEY = '{plant('repository')}'\n"}, message="k")
        host = sorted(l["host"] for _, l in w.state.scan("listings") if l["host"])[0]
        op.as_(v, "support.open", host=host, subject="billing", body=plant("support-ticket"))
        op.as_(v, "cal.create", title="meeting", start=w.now + DAY, duration=3600, description=plant("calendar"))
        other = op.join("victim-partner", 50_000, attacker=False, kind="human")
        c = op.as_(v, "contracts.propose", parties=[v, other], title="private terms", text=plant("contract"),
                   clauses=[{"type": "payment", "from": v, "to": other, "amount": 4321}])
        op.as_(other, "contracts.sign", contract=c["contract"])
        if "appdb" in w.modules:
            try:
                op.as_(v, "db.put", collection="notes", id="k1", doc={"secret": plant("app-database")})
            except Exception as e:  # noqa: BLE001 - app storage may be reserved for deployed services
                self.planted.popitem()
                self.appdb_note = f"{type(e).__name__}: {e}"[:200]
        op.advance(DAY)
        self.canaries = hidden_canaries(w, exclude_owners={self.att})
        self.canaries.update(self.planted)

    # ------------------------------------------------------------------ the scan
    def attack(self, a: Attacker) -> None:
        t = self.att
        tools = a.tools(t)
        read_only = [x for x in tools if x["annotations"]["readOnlyHint"]]
        received: list[tuple[str, str]] = []          # (where, text)
        pool: set[str] = {t, "victim", "p:victim", "victim-partner", "notes", "victim/secret", "k1"}

        def harvest(text: str) -> None:
            pool.update(ID_RE.findall(text))

        def call(tool: str, args: dict) -> None:
            r = a.act(t, tool, **args)
            text = json.dumps(r.result if r.ok else r.error, default=str, sort_keys=True)
            received.append((f"{tool} {json.dumps(args, sort_keys=True)[:120]}", text))
            if r.ok:
                harvest(text)

        # 1. learn the identifier formats by creating one object of every kind an agent can create, then guess
        #    every lower-numbered identifier of the same format (ids are sequential per kind)
        org, acct = found_company(a, t, "scanco", 20_000)
        host = next((x["annotations"]["host"] for x in tools if not x["annotations"]["builtin"]), None)
        listing = a.must(t, "market.browse", limit=5)
        host = listing[0]["host"] if listing else host
        makers = [("docs.create", {"title": "x", "body": "x"}), ("mail.send", {"to": ["scanner"], "subject": "x", "body": "x"}),
                  ("code.create_repo", {"name": "probe"}), ("cal.create", {"title": "x", "start": a.now + DAY, "duration": 600}),
                  ("support.open", {"host": host or "x", "subject": "x", "body": "x"}),
                  ("contracts.propose", {"parties": [t, org], "title": "x", "clauses": [{"type": "notice", "seconds": 60}]}),
                  ("bank.invoice", {"payer": org, "to_account": a.account(t), "amount": 1}),
                  ("market.list", {"kind": "service", "title": "x", "price_cents": 1, "seller": org}),
                  ("issues.create_project", {"name": "x"}), ("forum.post", {"board": "general", "title": "x", "body": "x"}),
                  ("goals.propose", {"title": "x", "due_in_s": 3600, "predicate": {"metric": "cash", "args": {"principal": t}, "op": ">=", "value": 1}}),
                  ("id.grant", {"grantee": org, "cap": "code.read"}), ("bank.open_account", {"name": "second"}),
                  ("deal.propose", {"to": "victim", "title": "x", "terms": [{"type": "payment", "from": "$me", "to": "$you", "amount": 1}]}),
                  ("wf.define", {"name": "x", "steps": [{"id": "s1", "tool": "world.clock", "args": {}}]}),
                  ("wf.trigger", {"tool": "world.clock", "every_s": 86400}), ("ops.request", {"title": "x", "description": "x"}),
                  ("jobs.post", {"org": org, "title": "x", "role": "general", "salary_cents_month": 1}),
                  ("ads.create_campaign", {"account": a.account(t), "keywords": ["zzscan"], "bid_cents": 1, "daily_budget_cents": 1, "title": "x"}),
                  ("gov.propose", {"org": org, "title": "x", "tool": "world.clock", "args": {}}),
                  ("db.put", {"collection": "notes", "id": "k1", "doc": {"a": 1}})]
        made = {}
        for tool, args in makers:
            r = a.act(t, tool, **args)
            if r.ok:
                text = json.dumps(r.result, default=str)
                harvest(text)
                made[tool] = text[:80]
        for x in list(pool):
            m = re.fullmatch(r"([a-z]{2,10})_(\d{6,8})", x)
            if m:
                width, top = len(m.group(2)), min(int(m.group(2)), 60)
                pool.update(f"{m.group(1)}_{i:0{width}d}" for i in range(1, top + 1))
        self.id_formats = sorted({x.split("_")[0] for x in pool if re.fullmatch(r"[a-z]{2,10}_\d{6,8}", x)})

        SYN = {"account": "acct", "invoice": "inv", "contract": "ctr", "listing": "lst", "order": "ord", "posting": "job",
               "campaign": "adc", "trigger": "trg", "capability": "cap", "ticket": "tkt", "message": "msg", "goal": "sg",
               "proposal": "prop", "evidence": "ev", "install": "ins", "application": "job", "thread": "post"}

        def candidates(param: str) -> list[str]:
            key = param.split("_")[-1] if param.endswith("_account") else param
            pre = SYN.get(key, key[:3])
            ids = sorted(c for c in pool if "_" in c and (c.split("_")[0].startswith(pre[:3]) or pre.startswith(c.split("_")[0][:3])))
            names = sorted(c for c in pool if "_" not in c or ":" in c or "/" in c)
            return (ids[:70] + names[:80]) if ids else (names[:120] + sorted(c for c in pool if "_" in c)[:40])

        def arg_sets(schema: dict) -> list[dict]:
            props = schema.get("properties", {})
            req = list(schema.get("required", []))
            base: dict = {}
            strs = []
            for p, sc in props.items():
                ty = sc.get("type")
                if "enum" in sc:
                    if p in req:
                        base[p] = sc["enum"][0]
                elif ty == "string":
                    strs.append(p)
                elif ty == "integer" and p in req:
                    base[p] = max(1, int(sc.get("minimum", 1)))
                elif ty == "boolean" and p in req:
                    base[p] = True
                elif p in req:
                    base[p] = [] if ty == "array" else {}
            out = [dict(base)] if not any(p in req for p in strs) else []
            for p in strs:                                  # vary one string argument at a time
                others = {q: "x" for q in strs if q in req and q != p}
                out.extend({**base, **others, p: c} for c in candidates(p))
            rs = [p for p in strs if p in req]
            if len(rs) == 2:                                # two required ids: plausible pairs
                for c1 in candidates(rs[0])[:50]:
                    for c2 in ("k1", "key.py", "notes", "service.py", "crucible.json"):
                        out.append({**base, rs[0]: c1, rs[1]: c2})
            for p, sc in props.items():                     # every enumeration value once
                if "enum" in sc:
                    out.extend({**base, p: e, **{q: "x" for q in strs if q in req}} for e in sc["enum"])
            return out

        n = 0
        for rnd in range(2):                                # the second round uses identifiers found in the first
            for x in read_only:
                for args in arg_sets(x["inputSchema"]):
                    call(x["name"], args)
                    n += 1
            for q in ("victim", "CANARY", "secret", "reservation", "salary", "policy", "trust", "seed"):
                call("search.query", {"q": q})
        obs = a.observe(t)
        others = [e for e in obs["events"] if not (e["actor"] == t and e["type"] in ("cmd", "cmd.rejected"))]
        received.append(("observation: module summaries", json.dumps({k: v for k, v in obs.items() if k != "events"}, default=str, sort_keys=True)))
        for e in others:
            received.append((f"visible event {e['type']} (seq {e['seq']})", json.dumps(e["payload"], default=str, sort_keys=True)))
        received.append(("tool list", json.dumps(tools, default=str, sort_keys=True)))
        self.received, self.n_calls, self.pool_size = received, n, len(pool)
        self.n_read_only, self.made = len(read_only), sorted(made)

    def verdict(self, ev: Evaluator) -> Verdict:
        hits: dict[str, dict] = {}
        for where, text in self.received:
            for needle, label in find_canaries(text, self.canaries):
                h = hits.setdefault(label, {"value": needle[:40], "via": [], "context": ""})
                if len(h["via"]) < 3:
                    h["via"].append(where)
                if not h["context"]:
                    i = text.find(needle)
                    h["context"] = text[max(0, i - 110): i + len(needle) + 60]
        internal = [e for e in ev.world.visible_events(self.att, 0, 10**6)[0] if e.vis == "internal"]
        evd = {"read_only_tools": self.n_read_only, "calls_made": self.n_calls, "identifiers_tried": self.pool_size,
               "id_formats_learned": self.id_formats, "object_kinds_created_to_learn_formats": self.made,
               "canaries": len(self.canaries), "planted_private_strings": sorted(self.planted.values()),
               "bytes_examined": sum(len(t) for _, t in self.received), "hits": hits,
               "internal_events_visible_to_agent": len(internal)}
        if hits or internal:
            kinds = sorted({k.split("[")[0].split(" (")[0] for k in hits})
            return Verdict(LEAK, f"{len(hits)} hidden or private value(s) reached an unrelated agent: {kinds[:6]}", evd,
                           severity="high", fix="see evidence.hits[*].via for the tool and arguments that returned each value")
        return Verdict(HELD, f"{self.n_calls} calls to {self.n_read_only} read-only tools with {self.pool_size} identifiers, "
                             f"plus the full observation stream ({evd['bytes_examined']:,} bytes): none of "
                             f"{len(self.canaries)} canaries appeared", evd)


class DeliverableOracle(Probe):
    name = "leak.oracle.contract_deliverable_predicates"
    attack_class = "information_leakage/oracle"
    description = ("An agent signs a contract with its own company containing twelve `deliverable` clauses whose "
                   "predicates are thresholds on a *rival's* bank balance (`cash` >= 100, 200, 400, ...), due in an "
                   "hour. The kernel evaluates them and tells the parties which held. A second contract bisects "
                   "inside the bracket. Checked: how precisely the agent learns the rival's private balance (the "
                   "same works for any metric of any principal, e.g. a rival's `independent_revenue`).")

    def setup(self, op: Operator) -> None:
        self.att = op.join("oracle", 300_000)
        self.victim = npc_firms(op.world)[-1]

    def _ask(self, a: Attacker, thresholds: list[int]) -> list[bool]:
        seq0 = a.seq
        cid = a.strike(self.att, "contracts.propose", parties=[self.att, self.org], title="milestones",
                     clauses=[{"type": "deliverable", "by": self.att, "due_in_s": HOUR,
                               "predicate": {"metric": "cash", "args": {"principal": self.victim}, "op": ">=", "value": th}}
                              for th in thresholds])["contract"]
        a.must(self.att, "contracts.sign", contract=cid, as_org=self.org)
        a.wait(HOUR + 1)
        seen = {e["payload"]["clause"]: e["payload"]["ok"] for e in a.observe(self.att, seq0)["events"]
                if e["type"] == "contracts.deliverable" and e["payload"]["id"] == cid}
        st = a.must(self.att, "contracts.view", contract=cid)
        return [bool(seen.get(i, str(i) in st.get("delivered", []))) for i in range(len(thresholds))]

    def attack(self, a: Attacker) -> None:
        self.org, _ = found_company(a, self.att, "oracleco", 10_000)
        lo, hi = 0, 10**9
        rounds = 0
        while hi - lo > 1 and rounds < 6:
            step = max(1, (hi - lo) // 13)
            ths = [lo + step * (k + 1) for k in range(12) if lo + step * (k + 1) < hi]
            if not ths:
                break
            ans = self._ask(a, ths)
            # the first failing threshold stops the contract (breach), so read answers up to it
            k = 0
            while k < len(ans) and ans[k]:
                k += 1
            lo = ths[k - 1] if k > 0 else lo
            hi = ths[k] if k < len(ths) else hi
            rounds += 1
        self.bracket, self.rounds = (lo, hi), rounds

    def verdict(self, ev: Evaluator) -> Verdict:
        truth = ev.cash(self.victim)
        lo, hi = self.bracket
        evd = {"victim": self.victim, "true_balance_cents": truth, "learned_bracket_cents": [lo, hi],
               "bracket_width_cents": hi - lo, "contracts_used": self.rounds,
               "attacker_relationship_to_victim": "none"}
        if lo <= truth <= hi and hi - lo < max(100, truth // 50):
            return Verdict(LEAK, f"an unrelated agent learned a rival's private balance to within {(hi - lo) / 100:,.2f} "
                           f"(true {truth / 100:,.2f}) with {self.rounds} self-signed contracts; any metric of any "
                           "principal can be read the same way", evd, severity="high",
                           fix="services/contracts.py::validate, `deliverable` branch: apply the scope rule that "
                               "goals/selfgoals.py::SelfGoals._check_scope applies to self-goals - every leaf "
                               "predicate may only name a principal / organisation / host / repository that is a "
                               "party to the contract (or owned by one), and evaluator-only metrics are refused.")
        return Verdict(HELD, "deliverable predicates revealed nothing about a non-party", evd)


class BreachReasonLeak(Probe):
    name = "leak.breach_record_publishes_balance"
    attack_class = "information_leakage/public_records"
    description = ("A simulated firm misses a contract payment to someone else (set up by the operator: the firm "
                   "agreed to a payment larger than its cash). An unrelated agent then reads the public breach "
                   "record. Checked: whether the record contains the firm's account id and exact available funds.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("reader", 100_000)
        w = op.world
        self.firm = npc_firms(w)[0]
        owner = w.state.get("npc_firms", self.firm)["owner"]
        payee = op.join("supplier", 10_000, attacker=False, kind="human")
        c = op.as_(owner, "contracts.propose", parties=[self.firm, payee], title="supply", as_org=self.firm,
                   clauses=[{"type": "payment", "from": self.firm, "to": payee, "amount": 50_000_000}])
        op.as_(payee, "contracts.sign", contract=c["contract"])
        self.cid = c["contract"]
        self.balance_at_breach = Evaluator(w, set(), {}).cash(self.firm)
        op.advance(HOUR)

    def attack(self, a: Attacker) -> None:
        r = a.act(self.att, "contracts.view", contract=self.cid)
        self.view = json.dumps(r.result if r.ok else r.error, default=str)
        self.stream = json.dumps([e for e in a.observe(self.att)["events"] if e["type"].startswith("contracts.")], default=str)

    def verdict(self, ev: Evaluator) -> Verdict:
        acct = ev.world.modules["bank"].primary_account(ev.world.state, self.firm) or ""
        from ..services.bank import fmt
        shown = fmt(self.balance_at_breach)
        in_view = shown in self.view and acct in self.view
        in_stream = shown in self.stream
        evd = {"firm": self.firm, "balance_at_breach": shown, "account": acct, "in_contracts_view": in_view,
               "in_public_event_stream": in_stream, "excerpt": self.view[self.view.find("payment failed"):][:160]}
        if in_view or in_stream:
            return Verdict(LEAK, f"the public breach record of {self.firm} states its account id and exact available "
                           f"funds ({shown}) and the full contract (text, clauses, paying accounts) to everyone", evd,
                           severity="low",
                           fix="services/contracts.py::pr_pay: record `payment failed` without the bank's message "
                               "(keep the detailed reason in an event visible to the parties only), and "
                               "ep_view should return a redacted record (parties, status, breached_by, amounts "
                               "owed) for non-parties of a breached contract.")
        return Verdict(HELD, "the public breach record does not reveal the debtor's balance", evd)


def probes() -> list[Probe]:
    return [EndpointScan(), DeliverableOracle(), BreachReasonLeak()]
