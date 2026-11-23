"""A continuous invariant monitor for long simulations (evaluator side; nothing here is a tool).

`InvariantMonitor(world).check()` recomputes, from raw state only, the properties a world must keep whatever its
participants do, and returns every violation it finds. It does not trust the modules' own `check_invariants`
(it runs those too and reports what they say under `module.<name>`): each law below is re-derived here from the
tables that hold the facts.

    ledger      every transaction's postings sum to zero; every posting has its mirror row and nothing else does;
                every balance equals the sum of its postings; all balances sum to zero; money held outside the
                system accounts equals the recorded net inflow from them (endowments, household income, fees,
                hardware, upkeep); no account is below its permitted floor
    contracts   states are legal and signed; the payments a contract records equal the ledger transactions that
                carry its reference, clause by clause; an unfinished payment obligation has a scheduled job;
                claims, invoices, direct-debit mandates and the two escrow accounts agree with what they stand for
    compute     running cores never exceed a provider's capacity; an instance does not use more than it has; a
                live service has a running instance; nothing keeps running unpaid beyond the cloud's grace
    companies   a bankrupt or dissolved company operates nothing; share registers add up
    scheduler   no job is overdue, every job is queued, no periodic process has disappeared
    inventory   stock, reserves and energy use are within physical bounds; goods are conserved; while attached,
                every change of industrial stock is explained by a recorded stock event
    processes   while attached: a scheduled process that fails repeatedly is reported whatever else looks healthy

`attach(every_s)` makes the monitor follow the world: it listens to committed events (to remember which event
posted which transaction, to keep the stock and process-failure tallies) and runs a full check whenever
`every_s` simulated seconds have passed. Every violation records what was violated, where, the sequence number
of the event that caused it when that is known (otherwise the head of the log when it was found), the simulated
time and the amounts involved.
"""
from __future__ import annotations

import re
import time as _time
from dataclasses import asdict, dataclass, field

DAY = 86400
CU_PER_CORE_HOUR = 100_000
SHARES = 1_000_000
CONTRACT_STATES = {"proposed", "active", "terminating", "completed", "breached", "terminated", "rejected"}
_CLAUSE = re.compile(r"\(clause (\d+)\)$")


@dataclass
class Violation:
    kind: str                 # what: the law that does not hold, e.g. "ledger.unbalanced_txn"
    where: str                # the record it concerns (account, transaction, contract, provider, job, ...)
    seq: int                  # sequence number of the causing event if known, else the log head at detection
    t: int                    # simulated time of detection
    amounts: dict = field(default_factory=dict)
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return f"{self.kind} at {self.where} (seq {self.seq}, t {self.t}): {self.detail} {self.amounts}"


class InvariantMonitor:
    def __init__(self, world, *, run_module_checks: bool = True, proc_failure_limit: int = 3,
                 slack_s: int = 2 * 3600, claim_slack_s: int = 2 * DAY):
        self.world = world
        self.run_module_checks = run_module_checks
        self.proc_failure_limit = int(proc_failure_limit)
        self.slack_s = int(slack_s)
        self.claim_slack_s = int(claim_slack_s)
        self.violations: list[Violation] = []        # first detection of every distinct (kind, where)
        self._seen: set[tuple[str, str]] = set()
        self.checks = 0
        self.wall_s = 0.0
        self.attached = False
        self.every_s: int | None = None
        self._last_check_t = world.now
        self._txn_seq: dict[str, int] = {}           # transaction id -> seq of the event that posted it
        self.proc_failures: dict[str, dict] = {}     # process name -> {count, first_seq, last_seq, last_error}
        self._stock_base: dict[str, int] | None = None
        self._stock_delta: dict[str, int] = {}
        self._stock_seq: dict[str, int] = {}
        self._periodic: dict[str, str] = {}
        self._out: list[Violation] = []

    # ------------------------------------------------------------------ following a world
    def attach(self, every_s: int = DAY) -> "InvariantMonitor":
        """Follow the world: tally events as they are committed and run `check()` every `every_s` simulated
        seconds. Returns self."""
        if self.attached:
            self.every_s = int(every_s)
            return self
        st = self.world.state
        self.every_s = int(every_s)
        self._last_check_t = self.world.now
        # module-level periodic processes are scheduled under their own name (key == process); jobs that belong to
        # a record (a trigger, a contract) may legitimately disappear with it and are not tracked
        self._periodic = {sid: j["proc"] for sid, j in st.tables["sched"].items() if j.get("period") and sid == j["proc"]}
        if "stock" in st.tables:
            self._stock_base = {k: int(s["qty"]) for k, s in st.tables["stock"].items()}
        self.world.listeners.append(self._on_event)
        self.attached = True
        return self

    def detach(self) -> None:
        if self.attached:
            try:
                self.world.listeners.remove(self._on_event)
            except ValueError:
                pass
            self.attached = False

    def _on_event(self, ev) -> None:
        t = ev.type
        if t == "bank.txn":
            self._txn_seq[ev.payload["id"]] = ev.seq
        elif t == "make.stock_add" or t == "make.stock_take":
            k = f"{ev.payload['fac']}|{ev.payload['sku']}"
            q = int(ev.payload["qty"])
            self._stock_delta[k] = self._stock_delta.get(k, 0) + (q if t == "make.stock_add" else -q)
            self._stock_seq[k] = ev.seq
        elif t == "proc.failed":
            p = ev.payload.get("proc", "?")
            r = self.proc_failures.setdefault(p, {"count": 0, "first_seq": ev.seq, "first_t": ev.t})
            r.update(count=r["count"] + 1, last_seq=ev.seq, last_t=ev.t, last_error=str(ev.payload.get("error", ""))[:200])
        elif t == "sched.add" and ev.payload.get("period") and ev.payload["id"] == ev.payload["proc"]:
            self._periodic[ev.payload["id"]] = ev.payload["proc"]
        if self.every_s is not None and t in ("time", "proc") and not self.world.state.in_tx \
                and self.world.now - self._last_check_t >= self.every_s:
            self.check()

    # ------------------------------------------------------------------ the check
    def check(self) -> list[Violation]:
        """Recompute every invariant from raw state. Returns the violations that hold right now; the first
        detection of each distinct one is also kept in `self.violations`."""
        t0 = _time.perf_counter()
        self._out = []
        self._last_check_t = self.world.now
        st = self.world.state
        facts = self._ledger(st) if "accounts" in st.tables else {}
        if "contracts" in st.tables:
            self._contracts(st, facts)
        if "accounts" in st.tables:
            self._money_instruments(st, facts)
        if "instances" in st.tables:
            self._compute(st, facts)
        if "companies" in st.tables:
            self._companies(st)
        self._scheduler(st)
        self._inventory(st)
        self._processes()
        if st.meta.get("seq") != self.world.seq or st.meta.get("hash") != self.world.last_hash:
            self._v("kernel.head_mismatch", "state.meta", "recorded head differs from the log head",
                    recorded_seq=st.meta.get("seq"), log_seq=self.world.seq)
        if self.run_module_checks and not st.in_tx:
            for msg in self.world.check_invariants():
                name, _, rest = msg.partition(": ")
                self._v(f"module.{name}", rest[:80], rest)
        for v in self._out:
            if (v.kind, v.where) not in self._seen:
                self._seen.add((v.kind, v.where))
                self.violations.append(v)
        self.checks += 1
        self.wall_s += _time.perf_counter() - t0
        return list(self._out)

    def _v(self, kind: str, where: str, detail: str = "", seq: int | None = None, **amounts) -> None:
        self._out.append(Violation(kind, str(where), int(self.world.seq if seq is None else seq), int(self.world.now),
                                   dict(amounts), detail))

    def _txseq(self, tid: str) -> int | None:
        return self._txn_seq.get(tid)

    # ------------------------------------------------------------------ ledger
    def _ledger(self, st) -> dict:
        accts, txns, rows = st.tables["accounts"], st.tables["txns"], st.tables["acct_txns"]
        system = {a for a, r in accts.items() if r["kind"] == "system"}
        n_post = 0
        ext = 0                               # net postings on the system accounts = -(money handed to the economy)
        ext_kind: dict[str, int] = {}
        pay: dict[tuple, list] = {}           # (contract, clause) -> [(amount, from account, txn)]
        last_bill: dict[str, int] = {}        # instance -> time of its last compute payment
        tx_amount: dict[str, dict] = {}
        for tid, tx in txns.items():
            s = 0
            for acct, d in tx["postings"]:
                s += d
                n_post += 1
                row = rows.get(f"{tid}|{acct}")
                if row is None:
                    self._v("ledger.posting_missing", f"{tid}|{acct}", "a transaction's posting has no ledger row",
                            self._txseq(tid), amount=d)
                elif row["amount"] != d or row["acct"] != acct:
                    self._v("ledger.posting_mismatch", f"{tid}|{acct}", "ledger row differs from its transaction",
                            self._txseq(tid), posting=d, row=row["amount"])
                if acct not in accts:
                    self._v("ledger.unknown_account", f"{tid}|{acct}", "posting to an account that does not exist",
                            self._txseq(tid), amount=d)
                elif acct in system:
                    ext += d
                    ext_kind[tx["kind"]] = ext_kind.get(tx["kind"], 0) + d
            if s != 0:
                self._v("ledger.unbalanced_txn", tid, "postings do not sum to zero", self._txseq(tid), sum=s,
                        postings=len(tx["postings"]))
            kind = tx["kind"]
            if kind == "contract" and tx["ref"]:
                m = _CLAUSE.search(tx["memo"])
                if m:
                    src = [a for a, d in tx["postings"] if d < 0]
                    amt = sum(d for _, d in tx["postings"] if d > 0)
                    pay.setdefault((tx["ref"], int(m.group(1))), []).append((amt, src[0] if src else None, tid))
            elif kind == "compute" and tx["memo"].startswith("instance "):
                iid = tx["memo"][9:]
                if tx["t"] > last_bill.get(iid, -1):
                    last_bill[iid] = tx["t"]
            elif kind == "invoice" and tx["ref"]:
                tx_amount[tid] = {"ref": tx["ref"], "postings": tx["postings"]}
        sums: dict[str, int] = {}
        for rid, row in rows.items():
            sums[row["acct"]] = sums.get(row["acct"], 0) + row["amount"]
        if len(rows) != n_post:               # some row belongs to no transaction posting: find which
            for rid, row in rows.items():
                tx = txns.get(row["txn"])
                if tx is None or not any(a == row["acct"] for a, _ in tx["postings"]):
                    self._v("ledger.orphan_posting", rid, "ledger row without a transaction posting",
                            self._txseq(row["txn"]), amount=row["amount"])
        total = private = 0
        for aid, a in accts.items():
            total += a["balance"]
            if aid not in system:
                private += a["balance"]
                if a["balance"] < -a["overdraft"]:
                    self._v("ledger.below_floor", aid, f"{a['kind']} account of {a['owner']} below its floor",
                            balance=a["balance"], floor=-a["overdraft"])
            if a["balance"] != sums.get(aid, 0):
                self._v("ledger.balance_mismatch", aid, "balance differs from the sum of its postings",
                        balance=a["balance"], postings=sums.get(aid, 0))
        if total != 0:
            self._v("ledger.not_conserved", "all accounts", "balances do not sum to zero", total=total)
        if private != -ext:
            self._v("ledger.external_mismatch", "private accounts",
                    "money outside the system accounts differs from the recorded net inflow from them",
                    private=private, recorded_net_inflow=-ext)
        self.money = {"private": private, "recorded_net_inflow": -ext,
                      "net_inflow_by_kind": {k: -v for k, v in sorted(ext_kind.items())},
                      "transactions": len(txns)}
        return {"pay": pay, "last_bill": last_bill, "invoice_tx": tx_amount}

    # ------------------------------------------------------------------ contracts, claims, invoices, escrow
    def _contracts(self, st, facts) -> None:
        pay = facts.get("pay", {})
        parties_idx = st.tables["contract_parties"]
        seen_pay = set()
        for cid, c in st.tables["contracts"].items():
            if c["status"] not in CONTRACT_STATES:
                self._v("contract.illegal_state", cid, f"unknown status {c['status']!r}")
                continue
            signed, parties = set(c["signed"]), set(c["parties"])
            if not signed <= parties:
                self._v("contract.foreign_signature", cid, "signed by a non-party", signers=sorted(signed - parties))
            if c["status"] in ("active", "terminating", "completed") and signed != parties:
                self._v("contract.unsigned_active", cid, f"{c['status']} without every party's signature",
                        missing=sorted(parties - signed))
            if c["status"] == "proposed" and signed == parties and len(parties) > 0:
                self._v("contract.signed_not_active", cid, "every party signed but the contract is not in force")
            for p in parties:
                if f"{cid}|{p}" not in parties_idx:
                    self._v("contract.party_index", f"{cid}|{p}", "party missing from the party index")
            for i, cl in enumerate(c["clauses"]):
                if cl["type"] != "payment":
                    continue
                n_rec = int(c["paid"].get(str(i), 0))
                led = pay.get((cid, i), [])
                seen_pay.add((cid, i))
                wrong = [a for a, _, _ in led if a != cl["amount"]]
                acct = c["accounts"].get(str(i))
                foreign = [s for _, s, _ in led if acct is not None and s != acct]
                if len(led) != n_rec or wrong or foreign:
                    self._v("contract.payments_mismatch", f"{cid}#{i}",
                            "payments recorded by the contract differ from the ledger",
                            self._txseq(led[-1][2]) if led else None, recorded=n_rec, ledger=len(led),
                            clause_amount=cl["amount"], ledger_total=sum(a for a, _, _ in led))
                total = 1 if not cl["every_s"] else cl["count"]
                if total is not None and n_rec > total:
                    self._v("contract.overpaid", f"{cid}#{i}", "more payments than the clause allows",
                            recorded=n_rec, allowed=total)
                unfinished = total is None or n_rec < total
                if c["status"] == "completed" and unfinished:
                    self._v("contract.completed_unpaid", f"{cid}#{i}", "completed with payments outstanding",
                            recorded=n_rec, due=total)
                if c["status"] in ("active", "terminating") and unfinished and not st.has("sched", f"pay|{cid}|{i}"):
                    self._v("contract.payment_unscheduled", f"{cid}#{i}",
                            "an outstanding payment has no scheduled job", recorded=n_rec, due=total,
                            amount=cl["amount"])
        for (cid, i), led in pay.items():
            if (cid, i) not in seen_pay and cid.startswith("ctr_"):
                self._v("contract.payment_without_contract", f"{cid}#{i}", "ledger payment for an unknown clause",
                        self._txseq(led[-1][2]), ledger=len(led), ledger_total=sum(a for a, _, _ in led))
        grace = int(self.world.spec.params.get("insolvency_grace_s", 14 * DAY))
        comps = st.tables.get("companies", {})
        for clid, cl in st.tables["claims"].items():
            ok = cl["amount"] > 0 and (
                (cl["status"] == "open" and cl.get("paid", 0) == 0) or
                (cl["status"] == "settled" and cl.get("paid", 0) == cl["amount"]) or
                (cl["status"] == "written_off" and 0 <= cl.get("paid", 0) <= cl["amount"]))
            if not ok:
                self._v("claim.inconsistent", clid, f"{cl['status']} claim with paid {cl.get('paid')}",
                        amount=cl["amount"], paid=cl.get("paid", 0))
            comp = comps.get(cl["debtor"])
            if cl["status"] == "open" and comp is not None and comp["status"] == "registered" \
                    and self.world.now - cl["t"] > grace + DAY + self.claim_slack_s:
                self._v("claim.unresolved_past_grace", clid,
                        f"{cl['debtor']} still trades with a claim unpaid beyond the insolvency grace",
                        amount=cl["amount"], age_s=self.world.now - cl["t"], grace_s=grace)

    def _money_instruments(self, st, facts) -> None:
        accts, txns = st.tables["accounts"], st.tables["txns"]
        for iid, inv in st.tables["invoices"].items():
            if inv["status"] != "paid":
                continue
            tx = txns.get(inv.get("txn") or "")
            if tx is None or tx["ref"] != iid or [inv["payee_acct"], inv["amount"]] not in [list(p) for p in tx["postings"]]:
                self._v("invoice.paid_without_payment", iid, "invoice marked paid without a matching transaction",
                        self._txseq(inv.get("txn") or ""), amount=inv["amount"])
        for mid, m in st.tables["mandates"].items():
            if m["used"] > m["max"]:
                self._v("mandate.over_limit", mid, "more was pulled than the mandate allows", used=m["used"],
                        max=m["max"])
        if "orders" in st.tables and "acct_escrow" in accts:
            held = sum(o["amount"] for o in st.tables["orders"].values() if o["status"] in ("paid", "delivered", "disputed"))
            if held != accts["acct_escrow"]["balance"]:
                self._v("escrow.mismatch", "acct_escrow", "market escrow differs from the open orders it holds for",
                        balance=accts["acct_escrow"]["balance"], open_orders=held)
        if "bounties" in st.tables and "acct_bounties" in accts:
            held = sum(b["reward"] for b in st.tables["bounties"].values() if b["status"] == "open")
            if held != accts["acct_bounties"]["balance"]:
                self._v("escrow.mismatch", "acct_bounties", "bounty escrow differs from the open rewards",
                        balance=accts["acct_bounties"]["balance"], open_rewards=held)

    # ------------------------------------------------------------------ compute
    def _compute(self, st, facts) -> None:
        now = self.world.now
        cloud = self.world.modules.get("cloud")
        grace = int(cloud.p("suspend_grace_s", 72 * 3600)) if cloud is not None else 72 * 3600
        insts = st.tables["instances"]
        used: dict[str, int] = {}
        orgs = st.tables.get("orgs", {})
        last_bill = facts.get("last_bill", {})
        hour = now // 3600
        for iid, i in insts.items():
            if i["status"] == "running":
                used[i["provider"]] = used.get(i["provider"], 0) + i["cores"]
                cap = i["cores"] * CU_PER_CORE_HOUR
                if i["hour"] == hour and i["used_cu"] > 1.5 * cap + 10_000:
                    self._v("compute.instance_overused", iid, "compute used this hour far above the instance's capacity",
                            used_cu=round(float(i["used_cu"]), 1), capacity_cu=cap)
                paid_at = max(i["created"], last_bill.get(iid, 0))
                if now - paid_at > 3600 + self.slack_s:
                    self._v("compute.unbilled_instance", iid, "running although its hourly bill has not been paid",
                            unpaid_for_s=now - paid_at, hourly_cents=i["price"] * i["cores"])
            elif i["status"] == "suspended" and i.get("suspended_at") is not None \
                    and now - i["suspended_at"] > grace + 3600 + self.slack_s:
                self._v("compute.unpaid_beyond_grace", iid, "suspended for non-payment longer than the cloud's grace",
                        suspended_for_s=now - i["suspended_at"], grace_s=grace)
            o = orgs.get(i["owner"])
            if i["status"] != "terminated" and o is not None and o["status"] != "active":
                self._v("company.defunct_operating", f"{i['owner']}:{iid}", f"{o['status']} organisation holds an instance",
                        cores=i["cores"])
        for pid, p in st.tables["providers"].items():
            if used.get(pid, 0) > p["cores"]:
                self._v("compute.oversubscribed", pid, "running cores exceed the provider's capacity",
                        used=used.get(pid, 0), capacity=p["cores"])
        for did, d in st.tables["deployments"].items():
            if d["status"] != "live":
                continue
            inst = insts.get(d["instance"])
            if inst is None or inst["status"] == "terminated":
                self._v("compute.live_on_dead_instance", did, f"service {d['host']} is live without a running instance",
                        instance=d["instance"])
            o = orgs.get(d["owner"])
            if o is not None and o["status"] != "active":
                self._v("company.defunct_operating", f"{d['owner']}:{d['host']}",
                        f"{o['status']} organisation still runs a service")

    # ------------------------------------------------------------------ companies
    def _companies(self, st) -> None:
        tot: dict[str, int] = {}
        for r in st.tables["shares"].values():
            tot[r["org"]] = tot.get(r["org"], 0) + r["n"]
            if r["n"] < 0:
                self._v("equity.negative_holding", f"{r['org']}|{r['holder']}", "negative shareholding", shares=r["n"])
        for org, n in tot.items():
            if n != SHARES:
                self._v("equity.shares_mismatch", org, "shares outstanding differ from the issue", shares=n,
                        issued=SHARES)
        orgs = st.tables.get("orgs", {})
        for org, c in st.tables["companies"].items():
            if c["status"] == "registered":
                continue
            o = orgs.get(org)
            if o is not None and o["status"] == "active":
                self._v("company.defunct_operating", org, f"{c['status']} company is still an active organisation")
            if "listings" in st.tables:
                n = sum(1 for lid in st.find_ids("listings", "seller", org)
                        if st.tables["listings"][lid]["status"] == "active")
                if n:
                    self._v("company.defunct_operating", f"{org}:listings", f"{c['status']} company still sells",
                            active_listings=n)
            open_accts = [a for a in st.find_ids("accounts", "owner", org) if st.tables["accounts"][a]["status"] == "open"]
            if open_accts:
                self._v("company.defunct_operating", f"{org}:accounts", f"{c['status']} company has open accounts",
                        accounts=len(open_accts),
                        balance=sum(st.tables["accounts"][a]["balance"] for a in open_accts))
            if "contract_parties" in st.tables:
                live = [r["contract"] for r in st.find("contract_parties", "party", org)
                        if st.tables["contracts"][r["contract"]]["status"] in ("active", "terminating")]
                if live:
                    self._v("company.defunct_operating", f"{org}:contracts", f"{c['status']} company has contracts in force",
                            contracts=len(live))

    # ------------------------------------------------------------------ scheduler
    def _scheduler(self, st) -> None:
        now = self.world.now
        sched = st.tables["sched"]
        queued = {(e[0], e[2], e[3]) for e in self.world._heap}
        for sid, j in sched.items():
            if j["at"] < now:
                self._v("sched.overdue", sid, f"{j['proc']} should have run", at=j["at"], now=now, late_s=now - j["at"])
            if (j["at"], j["n"], sid) not in queued:
                self._v("sched.unqueued", sid, f"{j['proc']} is scheduled but not in the run queue", at=j["at"])
            if j["proc"] not in self.world.processes:
                self._v("sched.unknown_process", sid, f"no process {j['proc']!r}")
        for sid, proc in self._periodic.items():
            if sid not in sched:
                self._v("sched.periodic_job_missing", sid, f"periodic process {proc} is no longer scheduled")

    # ------------------------------------------------------------------ inventory and physical bounds
    def _inventory(self, st) -> None:
        if "inventory" in st.tables:                     # market goods: conserved against what was minted
            stock: dict[str, int] = {}
            for k, r in st.tables["inventory"].items():
                stock[r["sku"]] = stock.get(r["sku"], 0) + r["qty"]
                if r["qty"] < 0:
                    self._v("inventory.negative", k, "negative goods holding", qty=r["qty"])
            for o in st.tables["orders"].values():
                if o["kind"] == "goods" and o["status"] in ("paid", "disputed") and not o.get("shipped"):
                    sku = st.tables["listings"][o["listing"]]["sku"]
                    stock[sku] = stock.get(sku, 0) + o["qty"]
            for sku in sorted(set(stock) | {k[13:] for k in st.meta if k.startswith("goods_minted:")}):
                minted = st.meta_get("goods_minted:" + sku, 0)
                if stock.get(sku, 0) != minted:
                    self._v("inventory.goods_not_conserved", sku, "goods in existence differ from goods minted",
                            existing=stock.get(sku, 0), minted=minted)
        if "stock" in st.tables:                         # industrial stock
            for k, s in st.tables["stock"].items():
                if s["qty"] < 0 or int(s["qty"]) != s["qty"]:
                    self._v("inventory.negative", k, "stock is negative or fractional", qty=s["qty"])
            for k, s in st.tables["sites"].items():
                if s["reserve"] < 0:
                    self._v("inventory.negative", f"site {k}", "more was extracted than the site held", reserve=s["reserve"])
            for k, g in st.tables["grid_days"].items():
                grid = (st.tables["regions"].get(g["region"]) or {}).get("grid")
                if grid and g["used"] > grid["capacity"] + 1e-6:
                    self._v("energy.over_capacity", k, "energy used exceeds the grid's daily capacity",
                            used=g["used"], capacity=grid["capacity"])
            if self._stock_base is not None:
                keys = set(self._stock_base) | set(self._stock_delta) | set(st.tables["stock"])
                for k in sorted(keys):
                    expect = self._stock_base.get(k, 0) + self._stock_delta.get(k, 0)
                    have = st.tables["stock"][k]["qty"] if k in st.tables["stock"] else 0
                    if expect != have:
                        self._v("inventory.unexplained_change", k, "stock changed without a recorded stock event",
                                self._stock_seq.get(k), expected=expect, actual=have)

    # ------------------------------------------------------------------ processes
    def _processes(self) -> None:
        for proc, r in sorted(self.proc_failures.items()):
            if r["count"] >= self.proc_failure_limit:
                self._v("process.repeated_failure", proc, r.get("last_error", ""), r["first_seq"], failures=r["count"],
                        first_t=r["first_t"], last_t=r.get("last_t"))

    # ------------------------------------------------------------------ reporting
    def summary(self) -> dict:
        by: dict[str, int] = {}
        for v in self.violations:
            by[v.kind] = by.get(v.kind, 0) + 1
        return {"checks": self.checks, "wall_s": round(self.wall_s, 3), "violations": len(self.violations),
                "by_kind": dict(sorted(by.items())), "process_failures": {k: dict(v) for k, v in sorted(self.proc_failures.items())},
                "money": getattr(self, "money", None)}


def check_world(world) -> list[Violation]:
    """One-off check of a world (no event tallies: stock-flow and process-failure checks need `attach`)."""
    return InvariantMonitor(world).check()


__all__ = ["InvariantMonitor", "Violation", "check_world"]
