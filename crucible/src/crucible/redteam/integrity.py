"""World-integrity probes: determinism of replay and regeneration after adversarial action sequences, isolation
of failing nested work, conservation, contract settlement, ordering of concurrent actors, crash recovery and
tamper detection of the event log."""
from __future__ import annotations

import copy
import os
import signal

import numpy as np

from ..core.canonical import canonical_bytes, chain_hash
from ..core.errors import InvariantViolation
from ..core.kernel import World
from .framework import (DAY, HELD, HOUR, INTEGRITY_BREAK, MONEY_CREATED, Attacker, Evaluator, NotApplicable, Operator,
                        Probe, Verdict, integrity_postcheck)
from .helpers import found_company, impossible_for

CODE_TOOLS = {"code.run", "code.test", "jobs.submit", "cloud.deploy", "caps.test", "net.fetch", "net.login", "caps.publish"}
NASTY_STR = ["", "x", "p:nobody", "o:none", "acct_000001", "acct_households", "acct_escrow", "acct_genesis", "sys",
             "../../etc/passwd", "a" * 300, "\u0000", "\ud800", "'; DROP TABLE events;--", "${input.x}", "$me", "$self",
             "admin.endow", "svc:x", "{}", "null", "0", "-1"]
NASTY_NUM = [0, -1, 1, 2, 10, 999, 10**6, 10**12, 10**12 + 1, 10**30, 1.5, True, None, "7", [1], {"a": 1}]


def fuzz(a: Attacker, actors: list[str], rng: np.random.Generator, n: int, ids: list[str], exclude: set[str] = CODE_TOOLS,
         advance: bool = True) -> dict:
    """A seeded barrage of in-world actions: schema-shaped calls with harvested ids, hostile strings and extreme
    numbers, malformed action objects, with random waits. Returns counts."""
    tools = [t for t in a.tools(actors[0]) if t["name"] not in exclude]
    ok = bad = 0

    def value(schema: dict, name: str):
        ty = schema.get("type")
        if isinstance(ty, list):
            ty = ty[0]
        r = rng.random()
        if "enum" in schema and r < 0.9:
            return schema["enum"][int(rng.integers(0, len(schema["enum"])))]
        if ty == "string":
            if r < 0.7 and ids:
                return ids[int(rng.integers(0, len(ids)))]
            return NASTY_STR[int(rng.integers(0, len(NASTY_STR)))]
        if ty in ("integer", "number"):
            if r < 0.6:
                return int(rng.integers(1, 200_000))
            return NASTY_NUM[int(rng.integers(0, len(NASTY_NUM)))]
        if ty == "boolean":
            return bool(rng.random() < 0.5)
        if ty == "array":
            return [value(schema.get("items", {"type": "string"}), name) for _ in range(int(rng.integers(0, 3)))]
        if ty == "object":
            if name == "clauses" or name == "terms":
                return clause()
            return {"k": NASTY_NUM[int(rng.integers(0, len(NASTY_NUM)))]}
        return NASTY_NUM[int(rng.integers(0, len(NASTY_NUM)))]

    def clause():
        p = [ids[int(rng.integers(0, len(ids)))] for _ in range(3)] if ids else ["x", "y", "z"]
        amt = NASTY_NUM[int(rng.integers(0, len(NASTY_NUM)))] if rng.random() < 0.4 else int(rng.integers(1, 50_000))
        kinds = [{"type": "payment", "from": p[0], "to": p[1], "amount": amt, "every_s": 3600, "count": 2},
                 {"type": "membership", "person": p[0], "org": p[1]},
                 {"type": "grant", "owner": p[0], "grantee": p[1], "cap": "bank.spend", "limit": amt},
                 {"type": "shares", "org": p[1], "from": p[0], "to": p[2], "shares": amt},
                 {"type": "deliverable", "by": p[0], "predicate": impossible_for(p[0]), "due_in_s": 3600,
                  "penalty": {"from": p[0], "to": p[1], "amount": amt}},
                 {"type": "notice", "seconds": amt}, {"type": "unknown"}, {}]
        return kinds[int(rng.integers(0, len(kinds)))]

    for i in range(n):
        actor = actors[int(rng.integers(0, len(actors)))]
        r = rng.random()
        if r < 0.04:
            res = a.raw(actor, [None, 5, "x", {"tool": 5}, {"tool": "bank.transfer"}, {"tool": "bank.transfer", "args": []},
                                {"tool": "world.clock", "args": {}, "idempotency_key": "k"},
                                {"args": {}}][int(rng.integers(0, 8))])
        else:
            t = tools[int(rng.integers(0, len(tools)))]
            props = t["inputSchema"].get("properties", {})
            req = set(t["inputSchema"].get("required", []))
            args = {k: value(s, k) for k, s in props.items() if k in req or rng.random() < 0.35}
            if "clauses" in props:
                args["clauses"] = [clause() for _ in range(int(rng.integers(1, 4)))]
            if "parties" in props and ids:
                args["parties"] = [actor] + [ids[int(rng.integers(0, len(ids)))] for _ in range(int(rng.integers(1, 3)))]
            key = f"k{int(rng.integers(0, 40))}" if rng.random() < 0.1 else None
            res = a.act(actor, t["name"], _key=key, **args)
        ok += res.ok
        bad += not res.ok
        if advance and rng.random() < 0.05:
            a.wait(int(rng.integers(1, 6 * HOUR)))
    return {"actions": n, "accepted": int(ok), "rejected": int(bad)}


def _seed_ids(a: Attacker, actors: list[str]) -> list[str]:
    """Identifiers an agent can see: itself, its accounts, its companies, public directories and listings."""
    ids = list(actors)
    for i, p in enumerate(actors):
        org, acct = found_company(a, p, f"fz{i}co", 300_000)
        ids += [org, acct, a.account(p) or ""]
    first = actors[0]
    for r in a.must(first, "id.directory", kind="orgs", limit=50) + a.must(first, "id.directory", kind="persons", limit=50):
        ids.append(r["id"])
    for l in a.must(first, "market.browse", limit=30):
        ids += [l["listing"], l["host"] or "", l["need"] or ""]
    return sorted({x for x in ids if x})


class ReplayAfterFuzz(Probe):
    name = "integrity.replay_and_regeneration_after_adversarial_sequence"
    attack_class = "world_integrity/determinism"
    world = {"store": "memory", "params": {"max_actions_per_hour": 0}}
    description = ("Three agents fire 1,500 seeded adversarial actions (schema-shaped calls with real identifiers, "
                   "hostile strings, extreme numbers, malformed action objects, replayed idempotency keys, crafted "
                   "contract clauses) interleaved with random waits, in a world backed by an event store. Checked: "
                   "replaying the stored log reproduces the state digest and head hash exactly; regenerating (re-"
                   "executing every recorded command and clock move through the handlers) reproduces every event "
                   "hash; invariants hold.")

    def setup(self, op: Operator) -> None:
        self.actors = [op.join(f"fz{i}", 2_000_000) for i in range(3)]

    def attack(self, a: Attacker) -> None:
        ids = _seed_ids(a, self.actors)
        self.counts = fuzz(a, self.actors, np.random.default_rng(20261122), 1500, ids)
        a.wait(2 * DAY)

    def verdict(self, ev: Evaluator) -> Verdict:
        w = ev.world
        w.flush()
        events = w.store.read_events(w.world_id)
        evd = dict(self.counts, events=len(events), sim_days=round(w.now / DAY, 2),
                   process_failures=sum(1 for e in events if e["type"] == "proc.failed"),
                   internal_errors=sum(1 for e in events if e["type"] == "cmd.rejected" and "internal error" in e["payload"]["error"]["message"]),
                   internal_error_examples=sorted({e["payload"]["tool"] + ": " + e["payload"]["error"]["message"][:110]
                                                   for e in events if e["type"] == "cmd.rejected"
                                                   and "internal error" in e["payload"]["error"]["message"]})[:8])
        try:
            rep = World.replay(w.spec, events, verify=True)
            evd["replay_exact"] = rep.digest() == w.digest() and rep.last_hash == w.last_hash
        except Exception as e:  # noqa: BLE001
            evd["replay_exact"], evd["replay_error"] = False, f"{type(e).__name__}: {e}"[:300]
        try:
            reg = World.regenerate(w.spec, events)
            evd["regeneration_exact"] = reg.last_hash == w.last_hash and reg.digest() == w.digest() and reg.seq == w.seq
            if not evd["regeneration_exact"]:
                regev = [e.to_dict() for e in reg.log]
                k = next((i for i, (x, y) in enumerate(zip(events, regev)) if x["hash"] != y["hash"]), min(len(events), len(regev)))
                evd["first_divergence"] = {"seq": k + 1, "original": events[k] if k < len(events) else None,
                                           "regenerated": regev[k] if k < len(regev) else None}
            reg.close()
        except Exception as e:  # noqa: BLE001
            evd["regeneration_exact"], evd["regeneration_error"] = False, f"{type(e).__name__}: {e}"[:300]
        evd["invariants"] = w.check_invariants()
        if not (evd["replay_exact"] and evd["regeneration_exact"]) or evd["invariants"]:
            return Verdict(INTEGRITY_BREAK, "the adversarial sequence broke replay, regeneration or an invariant", evd,
                           severity="critical")
        note = ""
        if evd["internal_errors"]:
            note = (f"; note: {evd['internal_errors']} action(s) ended in an unhandled exception inside a handler "
                    "(rolled back and reported as `internal error`; in a strict-mode harness these would raise)")
        return Verdict(HELD, f"{self.counts['actions']} adversarial actions ({self.counts['accepted']} accepted): replay "
                             f"and regeneration of {len(events)} events are exact" + note, evd)


class ConservationUnderFuzz(Probe):
    name = "integrity.conservation_under_adversarial_sequence"
    attack_class = "world_integrity/conservation"
    world = {"consumers": True, "params": {"max_actions_per_hour": 0}}
    slow = True
    description = ("The same kind of barrage (900 actions, three agents) in a world with paying simulated consumers, "
                   "run over three simulated days. Checked: the ledger sums to zero and recomputes from postings, no "
                   "non-source account is overdrawn, goods and escrow are conserved, and nothing moved from a "
                   "source account (genesis, external, fees) to the attackers.")

    def setup(self, op: Operator) -> None:
        self.actors = [op.join(f"cz{i}", 2_000_000) for i in range(3)]

    def attack(self, a: Attacker) -> None:
        ids = _seed_ids(a, self.actors)
        rng = np.random.default_rng(7)
        self.counts = {"actions": 0, "accepted": 0, "rejected": 0}
        for _ in range(6):
            c = fuzz(a, self.actors, rng, 150, ids, advance=False)
            for k in c:
                self.counts[k] += c[k]
            a.wait(12 * HOUR)

    def verdict(self, ev: Evaluator) -> Verdict:
        post = integrity_postcheck(ev.world, replay=False)
        supply = ev.world.modules["bank"].money_supply(ev.world)
        evd = dict(self.counts, postcheck=post, minted_to_attackers_cents=ev.minted_to(), money_supply=supply,
                   attacker_cash_gain_cents=ev.wealth_gain())
        if ev.minted_to():
            return Verdict(MONEY_CREATED, "money moved from a source account to the attackers", evd, severity="critical")
        if not post["ok"]:
            return Verdict(INTEGRITY_BREAK, "a conservation invariant failed after the barrage", evd, severity="critical")
        return Verdict(HELD, f"after {self.counts['actions']} adversarial actions and three days of demand: balances sum to "
                             "0, every balance equals its postings, escrow and goods invariants hold, nothing was minted", evd)


class NestedFailureIsolation(Probe):
    name = "integrity.failing_nested_call_leaves_nothing"
    attack_class = "world_integrity/transaction_isolation"
    description = ("A three-party contract is built so that its activation fails half-way: when the last party "
                   "signs, the membership clause and a grant clause are applied and then a share-transfer clause "
                   "fails (more shares than exist). A second case fails inside a nested bank move after an escrow "
                   "transfer (a bounty funded from an account that is emptied by an earlier clause in the same "
                   "command is not possible, so the case uses a marketplace order for more than the buyer has "
                   "after listing fees). Checked: every table is byte-identical before and after the failed "
                   "command, apart from the rejection record.")

    def setup(self, op: Operator) -> None:
        self.att, self.acc = op.join("ia", 800_000), op.join("ib", 200_000)

    def attack(self, a: Attacker) -> None:
        t, u = self.att, self.acc
        self.org, _ = found_company(a, t, "ico", 100_000)
        cid = a.must(t, "contracts.propose", parties=[t, self.org, u], title="package",
                     clauses=[{"type": "membership", "person": u, "org": self.org, "role": "admin"},
                              {"type": "grant", "owner": t, "grantee": u, "cap": "bank.spend"},
                              {"type": "payment", "from": t, "to": u, "amount": 50_000},
                              {"type": "shares", "org": self.org, "from": t, "to": u, "shares": 2_000_000}])["contract"]
        a.must(t, "contracts.sign", contract=cid, as_org=self.org)
        self.before = a._w.state.table_digests()            # evaluator-side snapshot (read-only)
        self.seq_before = a.seq
        self.sign = a.act(u, "contracts.sign", contract=cid)
        self.after = a._w.state.table_digests()
        self.cid = cid
        lst = a.must(t, "market.list", kind="service", title="big", price_cents=150_000, seller=self.org)["listing"]
        self.before2 = a._w.state.table_digests()
        self.buy = a.act(u, "market.buy", listing=lst, account=a.account(u), qty=2)
        self.after2 = a._w.state.table_digests()

    def verdict(self, ev: Evaluator) -> Verdict:
        st = ev.world.state
        changed = sorted(k for k in self.before if self.before[k] != self.after.get(k))
        changed2 = sorted(k for k in self.before2 if self.before2[k] != self.after2.get(k))
        c = st.get("contracts", self.cid)
        left = [e.type for e in ev.world.events_range(self.seq_before) if e.cmd is not None and e.type not in ("cmd.rejected",)][:5]
        evd = {"sign_error": self.sign.error, "tables_changed_by_failed_sign": changed,
               "contract_status": c["status"], "signed": c["signed"],
               "accomplice_is_member": st.has("members", f"{self.org}|{self.acc}"),
               "grants_to_accomplice": len(st.find("grants", "grantee", self.acc)),
               "buy_error": self.buy.error, "tables_changed_by_failed_buy": changed2,
               "escrow_cents": st.tables["accounts"]["acct_escrow"]["balance"]}
        if self.sign.ok or self.buy.ok:
            raise NotApplicable("the commands that were meant to fail succeeded; the probe does not test isolation")
        if changed or changed2 or evd["accomplice_is_member"] or evd["grants_to_accomplice"] or self.acc in c["signed"]:
            return Verdict(INTEGRITY_BREAK, f"a failed command left effects behind in {changed or changed2}", evd,
                           severity="critical")
        return Verdict(HELD, "both failing commands left every table byte-identical (membership, grant, signature and "
                             "escrow move were rolled back with the failed nested step)", evd)


class ContractSettlement(Probe):
    name = "integrity.contract_settlement_consistency"
    attack_class = "world_integrity/settlement"
    description = ("Five contracts between two colluders exercise every ending: paid in full, terminated by "
                   "notice mid-way, mutual termination, breach by an emptied account, and rejection after the "
                   "first signature. Checked from the ledger alone: payments made equal the contract's counter, "
                   "nothing is paid after an ending, a breach has exactly one claim for exactly one instalment, "
                   "and no payment job remains scheduled for an ended contract.")

    def setup(self, op: Operator) -> None:
        self.p, self.q = op.join("sa", 3_000_000), op.join("sb", 1_000_000)

    def attack(self, a: Attacker) -> None:
        p, q = self.p, self.q
        pay = lambda n, extra=(): [{"type": "payment", "from": p, "to": q, "amount": 10_000, "every_s": HOUR, "count": n},  # noqa: E731
                                   *extra]
        mk = lambda title, clauses: a.must(p, "contracts.propose", parties=[p, q], title=title, clauses=clauses)["contract"]  # noqa: E731
        self.c = {"full": mk("full", pay(4)), "notice": mk("notice", pay(50, [{"type": "notice", "seconds": 2 * HOUR}])),
                  "mutual": mk("mutual", pay(50)), "rejected": mk("rejected", pay(3))}
        for k in ("full", "notice", "mutual"):
            a.must(q, "contracts.sign", contract=self.c[k])
        a.must(q, "contracts.reject", contract=self.c["rejected"])
        a.wait(3 * HOUR)
        a.must(q, "contracts.terminate", contract=self.c["notice"])
        a.must(p, "contracts.terminate", contract=self.c["mutual"])
        a.must(q, "contracts.terminate", contract=self.c["mutual"])
        a.wait(6 * HOUR)
        # breach: a contract paid from a second account that is then emptied
        acct2 = a.must(p, "bank.open_account", name="ops")["account"]
        a.must(p, "bank.transfer", from_account=a.account(p), to=acct2, amount=25_000)
        cid = a.must(p, "contracts.propose", parties=[p, q], title="breach", clauses=pay(10))["contract"]
        a.act(q, "contracts.sign", contract=cid)
        self.c["breach"] = cid
        # re-sign with the small account is not possible after signing; instead drain the paying account
        a.wait(2 * HOUR)
        bal = a.observe(p)["bank"]["balances"][a.account(p)]
        a.must(p, "bank.transfer", from_account=a.account(p), to=acct2, amount=bal)
        a.wait(4 * HOUR)
        a.must(p, "bank.transfer", from_account=acct2, to=a.account(p), amount=bal)
        a.wait(6 * HOUR)

    def verdict(self, ev: Evaluator) -> Verdict:
        st = ev.world.state
        rows, problems = {}, []
        for k, cid in self.c.items():
            c = st.get("contracts", cid)
            tx = [e for e in ev.events("bank.txn") if e.payload["ref"] == cid and e.payload["kind"] == "contract"]
            claims = [dict(v) for _, v in st.scan("claims") if cid in v["source"]]
            ended_t = c.get("updated", 0)
            late = [e.seq for e in tx if c["status"] not in ("active", "terminating") and e.t > ended_t]
            jobs = [sid for sid, j in st.scan("sched") if j["args"].get("contract") == cid]
            rows[k] = {"status": c["status"], "payments": len(tx), "counter": c["paid"].get("0", 0), "claims": len(claims),
                       "paid_after_end": len(late), "jobs_left": len(jobs)}
            if len(tx) != c["paid"].get("0", 0):
                problems.append(f"{k}: {len(tx)} ledger payments vs counter {c['paid'].get('0', 0)}")
            if late:
                problems.append(f"{k}: paid after ending")
            if c["status"] not in ("active", "terminating") and jobs:
                problems.append(f"{k}: {len(jobs)} job(s) still scheduled")
        exp = {"full": ("completed", 4, 0), "rejected": ("rejected", 0, 0), "mutual": ("terminated", None, 0),
               "notice": ("terminated", None, 0), "breach": ("breached", None, 1)}
        for k, (status, n, n_claims) in exp.items():
            r = rows[k]
            if r["status"] != status or (n is not None and r["payments"] != n) or r["claims"] != n_claims:
                problems.append(f"{k}: expected {status}/{n}/{n_claims} claims, got {r}")
        evd = {"contracts": rows, "problems": problems, "group_cash_gain_cents": ev.wealth_gain()}
        if problems or ev.wealth_gain() > 0:
            return Verdict(INTEGRITY_BREAK, "contract settlement is inconsistent with the ledger: " + "; ".join(problems[:3]),
                           evd, severity="high")
        return Verdict(HELD, "for all five endings the ledger, the payment counters, the claims and the schedule agree", evd)


class _Racer:
    """A policy that tries, every time it wakes, to take 60% of a shared account for itself."""
    name, kind, version = "racer", "scripted", "1"
    batch_group = None

    def __init__(self, account: str, amount: int):
        self.account, self.amount, self.results = account, amount, []

    def reset(self, actor, rng):
        self.actor = actor

    def act(self, obs, tools):
        return {"tool": "bank.transfer", "args": {"from_account": self.account, "to": self.actor, "amount": self.amount}}

    def observe_result(self, action, result):
        self.results.append(bool(result["ok"]))

    def usage(self):
        return {}


def _race(world: World, racers: list[str], account: str, seed: int, until: int) -> dict:
    from ..agents.base import AsyncRunner
    pols = [_Racer(account, 600_000) for _ in racers]
    run = AsyncRunner(world, list(zip(racers, pols)), seed=seed, think_s=lambda actor, rng: 3600)
    for i in range(len(run.heap)):                      # everyone wakes at the same second, every hour
        run.heap[i] = (world.now + 3600, run.heap[i][1], run.heap[i][2])
    n = run.run_until(until)
    wins = {a: sum(p.results) for a, p in zip(racers, pols)}
    return {"actions": n, "wins": wins, "schedule_head": [(t, a) for t, a, _ in run.schedule[:8]],
            "digest": world.digest(), "balance": world.state.tables["accounts"][account]["balance"]}


class AsyncOrdering(Probe):
    name = "integrity.ordering_of_concurrent_actions"
    attack_class = "world_integrity/concurrency"
    description = ("Four co-owners of one company account (1,000,000 cents) run under the asynchronous runner and "
                   "all wake at the same simulated second, every hour for six hours, each trying to transfer 60% "
                   "of the account to itself. Checked: exactly one transfer can succeed in total and the account "
                   "never goes negative; the same seed gives the same winner, schedule and state digest (run again "
                   "on an identical fork); a different seed may pick another winner but never two.")

    def setup(self, op: Operator) -> None:
        self.racers = [op.join(f"race{i}", 100_000) for i in range(4)]
        first = self.racers[0]
        w = op.world
        op.as_(first, "id.create_org", handle="raceco", name="RaceCo", kind="company")
        op.as_(first, "biz.register", org="o:raceco", account=op.account(first))
        self.account = op.as_(first, "bank.open_account", owner="o:raceco")["account"]
        op.sys("admin.endow", principal=self.account, amount=1_000_000)
        for r in self.racers[1:]:
            inv = op.as_(first, "id.invite", org="o:raceco", person=r, role="owner")["invite"]
            op.as_(r, "id.respond_invite", invite=inv, accept=True)
        self.twin_same = w.fork("race-twin-same")
        self.twin_other = w.fork("race-twin-other")
        self.until = w.now + 6 * HOUR + 60

    def attack(self, a: Attacker) -> None:
        for r in self.racers:
            if r not in a.controlled:
                raise NotApplicable("racers must be attacker-controlled")
        self.main = _race(a._w, self.racers, self.account, seed=11, until=self.until)

    def verdict(self, ev: Evaluator) -> Verdict:
        same = _race(self.twin_same, self.racers, self.account, seed=11, until=self.until)
        other = _race(self.twin_other, self.racers, self.account, seed=12, until=self.until)
        for t in (self.twin_same, self.twin_other):
            t.close()
        evd = {"wins_seed11": self.main["wins"], "wins_seed11_again": same["wins"], "wins_seed12": other["wins"],
               "identical_digest_same_seed": self.main["digest"] == same["digest"],
               "identical_schedule_same_seed": self.main["schedule_head"] == same["schedule_head"],
               "account_balance_after": [self.main["balance"], same["balance"], other["balance"]],
               "actions": self.main["actions"]}
        one = all(sum(x["wins"].values()) == 1 and x["balance"] == 400_000 for x in (self.main, same, other))
        if not one or not evd["identical_digest_same_seed"] or not evd["identical_schedule_same_seed"]:
            return Verdict(INTEGRITY_BREAK, "simultaneous actions were not serialised consistently", evd, severity="critical")
        w11 = next(a for a, n in self.main["wins"].items() if n)
        w12 = next(a for a, n in other["wins"].items() if n)
        return Verdict(HELD, f"{self.main['actions']} simultaneous attempts: exactly one transfer succeeded in every run; the "
                             f"same seed reproduced winner ({w11}), schedule and state digest; with another seed the "
                             f"winner was {w12} and still exactly one", evd)


def _crash_child(path: str, world_id: str, conn) -> None:
    """Child process: load the world, act, flush, act again without flushing, die by SIGKILL."""
    import crucible
    crucible.load_all_modules()
    from ..storage.sqlite import SQLiteStore
    w = World.load(SQLiteStore(path), world_id)
    actor = "p:crash0"
    acct = w.modules["bank"].primary_account(w.state, actor)
    for i in range(20):
        w.execute(actor, {"tool": "bank.transfer", "args": {"from_account": acct, "to": "crash1", "amount": 100 + i}})
    w.advance_time(w.now + HOUR)
    w.flush()
    conn.send({"seq": w.seq, "hash": w.last_hash, "digest": w.digest()})
    for i in range(7):
        w.execute(actor, {"tool": "bank.transfer", "args": {"from_account": acct, "to": "crash1", "amount": 5}})
    conn.send({"unflushed_seq": w.seq})
    os.kill(os.getpid(), signal.SIGKILL)


class CrashRecovery(Probe):
    name = "integrity.recovery_after_crash"
    attack_class = "world_integrity/crash_recovery"
    world = {"store": "sqlite"}
    description = ("A world backed by an SQLite event store takes adversarial actions (including rejected and "
                   "failing ones), is snapshotted, and a separate writer process then loads it, acts, flushes, acts "
                   "again and is killed with SIGKILL before flushing. Checked: the store ends exactly at the last "
                   "flush; loading (snapshot + tail) and replaying from genesis agree with the state at that flush; "
                   "the recovered world keeps writing; the lost actions left no trace.")

    def setup(self, op: Operator) -> None:
        self.actors = [op.join(f"crash{i}", 500_000) for i in range(2)]

    def attack(self, a: Attacker) -> None:
        ids = _seed_ids(a, self.actors)
        self.counts = fuzz(a, self.actors, np.random.default_rng(3), 150, ids)

    def verdict(self, ev: Evaluator) -> Verdict:
        import multiprocessing as mp
        from ..storage.sqlite import SQLiteStore
        w = ev.world
        path, wid = w.store.path, w.world_id
        sid = w.snapshot()
        w.flush()
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_crash_child, args=(path, wid, child))
        proc.start()
        if not parent.poll(120):
            proc.kill()
            raise NotApplicable("the writer process did not start")
        info = parent.recv()
        lost = parent.recv() if parent.poll(60) else {"unflushed_seq": None}
        proc.join(60)
        s2 = SQLiteStore(path)
        w2 = World.load(s2, wid)
        full = World.replay(w.spec, s2.read_events(wid))
        evd = {"fuzz": self.counts, "snapshot": sid, "writer_exit": proc.exitcode, "seq_at_last_flush": info["seq"],
               "seq_in_memory_at_kill": lost["unflushed_seq"], "store_last_seq": s2.last_seq(wid),
               "loaded_equals_flush": (w2.digest(), w2.last_hash, w2.seq) == (info["digest"], info["hash"], info["seq"]),
               "genesis_replay_equals_flush": (full.digest(), full.last_hash) == (info["digest"], info["hash"]),
               "invariants_after_recovery": w2.check_invariants(),
               "ledger_after_recovery": w2.modules["bank"].check_ledger_consistency(w2)}
        acct = w2.modules["bank"].primary_account(w2.state, "p:crash1")
        r = w2.execute("p:crash1", {"tool": "bank.transfer", "args": {"from_account": acct, "to": "crash0", "amount": 1}})
        try:
            w2.flush()
            evd["continues_writing"] = bool(r.ok) and s2.last_seq(wid) == w2.seq
        except Exception as e:  # noqa: BLE001
            evd["continues_writing"], evd["write_error"] = False, f"{type(e).__name__}: {e}"[:200]
        ok = (proc.exitcode == -signal.SIGKILL and evd["store_last_seq"] == info["seq"] and evd["loaded_equals_flush"]
              and evd["genesis_replay_equals_flush"] and not evd["invariants_after_recovery"]
              and not evd["ledger_after_recovery"] and evd["continues_writing"]
              and (lost["unflushed_seq"] or 0) > info["seq"])
        ev.world.store = None            # the probe's own handle must not flush over the recovered writer
        ev.world._unflushed = []
        if not ok:
            return Verdict(INTEGRITY_BREAK, "recovery after a killed writer did not reproduce the last durable state", evd,
                           severity="critical")
        return Verdict(HELD, f"writer killed with {lost['unflushed_seq'] - info['seq']} unflushed events: the store ends at "
                             "the last flush; snapshot+tail load and genesis replay both equal that state; writing resumed", evd)


class TamperDetection(Probe):
    name = "integrity.event_log_tamper_detection"
    attack_class = "world_integrity/tamper_evidence"
    world = {"store": "memory"}
    description = ("After a short adversarial run the stored event log is altered in five ways by someone with "
                   "write access to the store (not an in-world power): change the amount in one payment event; "
                   "the same and recompute that event's own hash; delete an event; swap two events; change the "
                   "visibility of an internal event to public. Checked: verification on load fails each time and "
                   "names the place.")

    def setup(self, op: Operator) -> None:
        self.actors = [op.join(f"tam{i}", 500_000) for i in range(2)]

    def attack(self, a: Attacker) -> None:
        p, q = self.actors
        for i in range(12):
            a.act(p, "bank.transfer", from_account=a.account(p), to=q, amount=1000 + i)
        a.wait(HOUR)
        for i in range(12):
            a.act(q, "bank.transfer", from_account=a.account(q), to=p, amount=500 + i)

    @staticmethod
    def _verify(spec, events) -> str:
        try:
            World.replay(spec, events, verify=True)
            return "ACCEPTED"
        except InvariantViolation as e:
            return f"detected: {e}"
        except Exception as e:  # noqa: BLE001
            return f"detected: {type(e).__name__}: {e}"[:160]

    def verdict(self, ev: Evaluator) -> Verdict:
        w = ev.world
        w.flush()
        orig = w.store.read_events(w.world_id)
        k = next(i for i, e in enumerate(orig) if e["type"] == "bank.txn" and e["payload"]["kind"] == "transfer")
        res = {"untouched": self._verify(w.spec, copy.deepcopy(orig))}
        t = copy.deepcopy(orig)
        t[k]["payload"]["postings"][0][1] -= 10**6
        t[k]["payload"]["postings"][1][1] += 10**6
        res["amount_changed"] = self._verify(w.spec, t)
        prev = t[k - 1]["hash"]
        body = {x: t[k][x] for x in ("seq", "t", "type", "actor", "cmd", "payload", "vis")}
        t[k]["hash"] = chain_hash(prev, canonical_bytes(body))
        res["amount_changed_and_own_hash_recomputed"] = self._verify(w.spec, t)
        t = copy.deepcopy(orig)
        del t[k]
        res["event_deleted"] = self._verify(w.spec, t)
        t = copy.deepcopy(orig)
        t[k], t[k + 1] = t[k + 1], t[k]
        res["events_swapped"] = self._verify(w.spec, t)
        t = copy.deepcopy(orig)
        j = next(i for i, e in enumerate(t) if e["vis"] == "internal" and i > 10)
        t[j]["vis"] = "public"
        res["visibility_changed"] = self._verify(w.spec, t)
        evd = {"events": len(orig), "tampered_seq": orig[k]["seq"], "verification": res}
        if res["untouched"] != "ACCEPTED" or any(v == "ACCEPTED" for name, v in res.items() if name != "untouched"):
            return Verdict(INTEGRITY_BREAK, "an altered event log passed verification", evd, severity="critical")
        return Verdict(HELD, "all five alterations of a stored event were detected by hash-chain verification on load", evd)


class SuffixRewrite(Probe):
    name = "integrity.event_log_suffix_rewrite_needs_anchor"
    attack_class = "world_integrity/tamper_evidence"
    world = {"store": "memory"}
    description = ("Limits of the hash chain against someone with write access to the store (not an in-world "
                   "power): change one payment event and recompute the hash of every later event; and simply drop "
                   "the last events. Checked: whether `World.replay(verify=True)` / `World.load` notice, and what "
                   "does (comparison with a head hash recorded elsewhere).")

    def setup(self, op: Operator) -> None:
        self.actors = [op.join(f"sfx{i}", 500_000) for i in range(2)]

    def attack(self, a: Attacker) -> None:
        p, q = self.actors
        for i in range(10):
            a.act(p, "bank.transfer", from_account=a.account(p), to=q, amount=1000 + i)

    def verdict(self, ev: Evaluator) -> Verdict:
        w = ev.world
        w.flush()
        orig = w.store.read_events(w.world_id)
        head = w.last_hash
        k = next(i for i, e in enumerate(orig) if e["type"] == "bank.txn" and e["payload"]["kind"] == "transfer")
        t = copy.deepcopy(orig)
        t[k]["payload"]["postings"][0][1] -= 77
        t[k]["payload"]["postings"][1][1] += 77
        h = t[k - 1]["hash"]
        for e in t[k:]:
            h = chain_hash(h, canonical_bytes({x: e[x] for x in ("seq", "t", "type", "actor", "cmd", "payload", "vis")}))
            e["hash"] = h
        res = {}
        try:
            r = World.replay(w.spec, t, verify=True)
            res["suffix_rewritten"] = {"replay_verify": "ACCEPTED", "head_hash_differs_from_recorded_head": r.last_hash != head,
                                       "invariants": r.check_invariants()}
        except Exception as e:  # noqa: BLE001
            res["suffix_rewritten"] = {"replay_verify": f"detected: {e}"[:120]}
        try:
            r = World.replay(w.spec, copy.deepcopy(orig[:-6]), verify=True)
            res["tail_truncated"] = {"replay_verify": "ACCEPTED", "head_hash_differs_from_recorded_head": r.last_hash != head}
        except Exception as e:  # noqa: BLE001
            res["tail_truncated"] = {"replay_verify": f"detected: {e}"[:120]}
        evd = {"events": len(orig), "recorded_head_hash": head, **res,
               "also": "snapshots are zlib-compressed pickles (World.from_blob -> pickle.loads): whoever can write "
                       "the snapshot table can run code in the operator's process when the world is loaded"}
        if any(v.get("replay_verify") == "ACCEPTED" for v in res.values()):
            return Verdict(INTEGRITY_BREAK, "store-level adversary only (no in-world route): a rewritten suffix and a "
                           "truncated tail both pass chain verification; they are detectable only by comparing the "
                           "head hash with a copy kept outside the store, which nothing does today", evd, severity="low",
                           fix="Keep the head (seq, hash) of every world outside the event table and check it on "
                               "load: e.g. write it into each experiment's results file and into the snapshot row, "
                               "and have World.load / `crucible world replay --verify` compare (the store already "
                               "records a hash per snapshot; `World.load` does not compare the replayed tail's "
                               "starting point with it). Sign or HMAC snapshot blobs, or replace pickle with the "
                               "canonical JSON state dump, before stores are shared between machines.")
        return Verdict(HELD, "suffix rewrite and truncation are detected", evd)


def probes() -> list[Probe]:
    return [ReplayAfterFuzz(), ConservationUnderFuzz(), NestedFailureIsolation(), ContractSettlement(), AsyncOrdering(),
            CrashRecovery(), TamperDetection(), SuffixRewrite()]
