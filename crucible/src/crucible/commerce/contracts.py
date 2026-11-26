"""Escrowed commitments with independent negotiation, disputes and resolution (module `commit`, host `commit`).

The 4.0 `contracts` service enforces agreements by scheduled debits and public breach records: a payer who has no
money when a payment falls due breaches, and the creditor is left with a claim. A commitment is the complementary
instrument for work paid on delivery: the payer's money is moved into escrow (`acct_commit_escrow`) when the
agreement is made, so the payee knows it exists, and it leaves escrow only by the agreement's own rules.

* **Negotiation** (`commit.offer`, `commit.respond` accept | reject | counter, `commit.withdraw`, `commit.thread`):
  offers and counter-offers in a persistent thread, written with `$me` / `$you`; nothing binds until the other side
  accepts the latest terms. Simulated participants decide through the `on_commit_offer` hook from their own state.
* **Terms**: a payer, 1-8 milestones (amount, deliverable, due), an optional arbiter (a third principal unrelated to
  either party) with a fee, a review window, a dispute window and a default split for disputes nobody resolves.
* **Deliverables**: `acceptance` (the payee delivers; the payer accepts, disputes, or stays silent - silence past the
  review window releases the money), `predicate` (a metrics predicate over the parties' own data, checked by the
  world at the due time - see goals/metrics.py), `task` (an `inst` project task: done, by the payee, at a minimum
  measured quality).
* **Enforcement**: money moves only through `bank`: escrow -> payee on release, escrow -> payer on refund. A payee
  who misses a due date breaches (public record) and the payer is refunded; a payer cannot breach (its money is in
  escrow).
* **Disputes and resolution**: a disputed milestone is frozen until (1) both parties post the same split
  (`commit.settle`: a negotiated settlement), (2) the arbiter rules (`commit.rule`, paid its fee from the disputed
  amount), or (3) the dispute window closes and the agreed default split applies.
* **Public record** (`commit.record`): counts of commitments completed, breaches, disputes raised and how they
  ended - never amounts, accounts or balances.
* **One contract record** (wave 3b, SP-2.3 contract migration; `primitives.OWNERS["contract"]` is
  `contracts.contracts`): where the 4.0 `contracts` module is installed, every commitment IS a contract: when it is
  made, this module creates the `contracts` record of the agreement (parties, title, an `escrow` clause naming the
  commitment), signed by both sides (they agreed by accepting the terms), and ends it when the commitment closes
  (`completed`, or `breached` by the payee that missed a due date) or is cancelled (`terminated`). The commitment
  rows keep only the escrow details (milestones, deliverables, disputes) of that contract.
"""
from __future__ import annotations

import numbers

from ..core import schema as S
from ..core.errors import Conflict, CrucibleError, Forbidden, InvalidAction, NotFound, contain
from ..core.kernel import SYSTEM, Ctx, Module, applier, endpoint, process, register_module
from ..services.identity import ident

DAY = 86400
ESCROW = "acct_commit_escrow"
MAX_AMOUNT = 10**12
MAX_MILESTONES = 8
TERMS = {"type": "object", "additionalProperties": True}
TERMINAL = ("released", "refunded", "resolved")


def _int(x, lo, hi, what) -> int:
    if isinstance(x, bool) or not isinstance(x, numbers.Integral):
        raise InvalidAction(f"{what} must be an integer")
    if not lo <= int(x) <= hi:
        raise InvalidAction(f"{what} must be between {lo} and {hi}")
    return int(x)


def _subst(x, me, you):
    if isinstance(x, str):
        return me if x == "$me" else you if x == "$you" else x
    if isinstance(x, dict):
        return {k: _subst(v, me, you) for k, v in x.items()}
    if isinstance(x, list):
        return [_subst(v, me, you) for v in x]
    return x


@register_module
class Commitments(Module):
    name = "commit"
    depends = ("identity", "bank")

    def install(self, world):
        st = world.state
        st.declare_table("commit_offers", index=("thread", "from", "to", "status"))
        st.declare_table("commitments", index=("payer", "payee", "status", "arbiter"))
        world.hooks["commit"] = self

    def genesis(self, ctx):
        ctx.emit("bank.account", {"id": ESCROW, "owner": SYSTEM, "kind": "escrow", "name": "commitment escrow",
                                  "overdraft": 0}, vis="internal")
        ctx.schedule(ctx.now + 3600, "commit.expire", {}, prio=57, period=3600, key="commit.expire")

    # ================================================================== validation
    def concrete(self, ctx: Ctx, terms, me: str, you: str) -> dict:
        """Concrete, validated terms (placeholders replaced). Raises if anything could not be executed or would
        reach beyond the two parties."""
        from ..goals.metrics import check_predicate, check_predicate_scope, is_related
        st = ctx.state
        if not isinstance(terms, dict) or set(terms) - {"title", "payer", "milestones", "arbiter", "arbiter_fee_cents",
                                                        "review_s", "dispute_s", "default_split"}:
            raise InvalidAction("terms: title, payer, milestones, arbiter, arbiter_fee_cents, review_s, dispute_s, "
                                "default_split")
        t = _subst(terms, me, you)
        payer = t.get("payer", "$me")
        payer = me if payer in ("$me", me) else you if payer in ("$you", you) else None
        if payer is None:
            raise InvalidAction("payer must be $me or $you")
        payee = you if payer == me else me
        ms_in = t.get("milestones")
        if not isinstance(ms_in, list) or not 1 <= len(ms_in) <= MAX_MILESTONES:
            raise InvalidAction(f"1-{MAX_MILESTONES} milestones")
        ms = []
        for i, m in enumerate(ms_in):
            if not isinstance(m, dict) or set(m) - {"amount_cents", "deliverable", "due_in_s"}:
                raise InvalidAction(f"milestone {i}: amount_cents, deliverable, due_in_s")
            amount = _int(m.get("amount_cents"), 1, MAX_AMOUNT, f"milestone {i}: amount_cents")
            due = _int(m.get("due_in_s", 7 * DAY), 3600, 366 * DAY, f"milestone {i}: due_in_s")
            d = m.get("deliverable") or {"kind": "acceptance"}
            if not isinstance(d, dict):
                raise InvalidAction(f"milestone {i}: deliverable is an object")
            kind = d.get("kind")
            if kind == "acceptance":
                dv = {"kind": kind, "what": str(d.get("what", ""))[:500]}
            elif kind == "predicate":
                errs = check_predicate(d.get("predicate"))
                if errs:
                    raise InvalidAction(f"milestone {i}: " + "; ".join(errs[:3]))
                from .observables import in_world_errors
                errs = in_world_errors(ctx.world, d["predicate"], [payer, payee])
                if errs:
                    raise Forbidden(f"milestone {i}: " + "; ".join(errs[:3]))
                dv = {"kind": kind, "predicate": d["predicate"]}
            elif kind == "task":
                tk = st.get("inst_tasks", d.get("task")) if "inst_tasks" in st.tables and \
                    isinstance(d.get("task"), str) else None
                if tk is None or tk["status"] in ("done", "cancelled"):
                    raise InvalidAction(f"milestone {i}: an open inst project task")
                prj = st.tables["inst_projects"][tk["project"]]
                partners = {r["partner"] for r in st.find("inst_partners", "project", tk["project"])
                            if r["status"] == "joined"}
                if payer not in partners and payer != prj["lead"]:
                    raise Forbidden(f"milestone {i}: the payer must own or be a partner of the task's project")
                q = d.get("min_quality", 0.0)
                if isinstance(q, bool) or not isinstance(q, numbers.Real) or not 0.0 <= float(q) <= 1.0:
                    raise InvalidAction(f"milestone {i}: min_quality in [0, 1]")
                dv = {"kind": kind, "task": d["task"], "min_quality": float(q)}
            else:
                raise InvalidAction(f"milestone {i}: deliverable kind acceptance, predicate or task")
            ms.append({"amount": amount, "deliverable": dv, "due_in_s": due})
        arbiter = t.get("arbiter")
        fee = 0
        if arbiter:
            arbiter = ident(ctx.world).resolve_handle(st, arbiter)
            if arbiter in (payer, payee) or is_related(ctx.world, payer, arbiter) or \
                    is_related(ctx.world, payee, arbiter) or is_related(ctx.world, arbiter, payer) or \
                    is_related(ctx.world, arbiter, payee):
                raise Forbidden("an arbiter must be independent of both parties")
            fee = _int(t.get("arbiter_fee_cents", 0), 0, MAX_AMOUNT, "arbiter_fee_cents")
            if fee * 10 > min(m["amount"] for m in ms):
                raise InvalidAction("the arbiter's fee is at most 10% of the smallest milestone")
        else:
            arbiter = None
        ds = t.get("default_split", 0.5)
        if isinstance(ds, bool) or not isinstance(ds, numbers.Real) or not 0.0 <= float(ds) <= 1.0:
            raise InvalidAction("default_split in [0, 1] (the payee's share when a dispute is not resolved)")
        return {"title": str(t.get("title") or "commitment")[:120], "payer": payer, "payee": payee,
                "milestones": ms, "arbiter": arbiter, "arbiter_fee": fee,
                "review_s": _int(t.get("review_s", 3 * DAY), 3600, 60 * DAY, "review_s"),
                "dispute_s": _int(t.get("dispute_s", 7 * DAY), 3600, 90 * DAY, "dispute_s"),
                "default_split": round(float(ds), 4)}

    def _payer_account(self, ctx: Ctx, payer: str, account: str | None) -> str:
        """The payer's account for the deposit. Committing money to escrow needs full authority over the account
        (a spending grant with a monthly limit cannot commit it)."""
        bank = ctx.world.modules["bank"]
        st = ctx.state
        a = account or bank.primary_account(st, payer)
        if a is None or not st.has("accounts", a) or bank.account_owner(st, a) != payer \
                or st.tables["accounts"][a]["status"] != "open":
            raise InvalidAction("name an open account of the payer")
        if ident(ctx.world).authorize_standing(ctx, payer, a) is not None:
            raise Forbidden("a spending grant with a limit cannot commit money to escrow")
        return a

    def party(self, ctx, as_org) -> str:
        if not as_org:
            return ctx.actor
        org = ident(ctx.world).resolve_handle(ctx.state, as_org)
        ident(ctx.world).authorize(ctx, org, "org.admin")
        return org

    # ================================================================== negotiation
    def offer(self, ctx: Ctx, me: str, you: str, terms: dict, message: str = "", account: str | None = None,
              expires_in_s: int = 3 * DAY, parent: str | None = None) -> str:
        st = ctx.state
        c = self.concrete(ctx, terms, me, you)
        acct = self._payer_account(ctx, me, account) if c["payer"] == me else None
        oid = st.new_id("coffer")
        p = st.get("commit_offers", parent) if parent else None
        ctx.emit("commit.offer", {"id": oid, "from": me, "to": you, "terms": terms, "concrete": c,
                                  "message": str(message)[:4000], "account": acct, "expires": ctx.now + expires_in_s,
                                  "thread": p["thread"] if p else oid, "round": (p["round"] + 1) if p else 0,
                                  "parent": parent, "by": ctx.actor}, vis=(me, you))
        hook = ctx.world.hooks.get("on_commit_offer")
        if hook:
            hook(ctx, oid, you)
        return oid

    def accept(self, ctx: Ctx, oid: str, who: str, account: str | None = None) -> str:
        st = ctx.state
        o = st.tables["commit_offers"][oid]
        if ctx.now > o["expires"]:
            raise Conflict("offer expired")
        c = self.concrete(ctx, o["terms"], o["from"], o["to"])     # terms re-checked against the world as it is now
        if c["payer"] == o["from"]:
            acct = o["account"]
            a = st.get("accounts", acct) if acct else None
            if a is None or a["status"] != "open" or a["owner"] != o["from"]:
                raise Conflict("the offering payer's account is no longer available")
        else:
            acct = self._payer_account(ctx, who, account)
        total = sum(m["amount"] for m in c["milestones"])
        bank = ctx.world.modules["bank"]
        cid = st.new_id("cmt")
        bank.move(ctx, acct, ESCROW, total, "escrow", f"commitment {cid}: {c['title']}", ref=cid)
        ms = [{"amount": m["amount"], "deliverable": m["deliverable"], "due": ctx.now + m["due_in_s"],
               "status": "pending", "delivered_t": None, "note": "", "dispute": None, "outcome": None,
               "proposals": {}} for m in c["milestones"]]
        ctx.emit("commit.created", {"id": cid, "offer": oid, "thread": o["thread"], "payer": c["payer"],
                                    "payee": c["payee"], "arbiter": c["arbiter"], "arbiter_fee": c["arbiter_fee"],
                                    "title": c["title"], "milestones": ms, "account": acct, "review_s": c["review_s"],
                                    "dispute_s": c["dispute_s"], "default_split": c["default_split"],
                                    "held": total}, vis=self._vis(c))
        ctx.emit("commit.offer_status", {"id": oid, "status": "accepted", "commitment": cid}, vis=(o["from"], o["to"]))
        self._open_contract(ctx, cid, c)
        for i, m in enumerate(ms):
            proc = "commit.check" if m["deliverable"]["kind"] == "predicate" else "commit.due"
            ctx.schedule(m["due"], proc, {"commitment": cid, "milestone": i}, prio=36, key=f"cmt|{cid}|{i}")
        return cid

    def _open_contract(self, ctx: Ctx, cid: str, c: dict) -> str | None:
        """The `contracts` record of a new commitment (module docstring); None without the contracts module."""
        contracts = ctx.world.modules.get("contracts")
        if contracts is None:
            return None
        parties = sorted({c["payer"], c["payee"]})

        def make(sub):
            sub.escrow_instrument = "commit"
            return contracts.propose(sub, c["payer"], parties,
                                     [{"type": "escrow", "instrument": "commit", "ref": cid}],
                                     str(c["title"])[:120], f"escrowed commitment {cid}", 3600, tags=("commitment",))
        k = ctx.run_as(c["payer"], make)
        for p in parties:
            ctx.run_as(p, lambda sub, p=p: contracts.sign(sub, k, p, None))
        ctx.emit("commit.contract", {"id": cid, "contract": k}, vis=tuple(parties))
        return k

    def _end_contract(self, ctx: Ctx, cid: str, status: str, reason: str, by: str | None = None) -> None:
        c = ctx.state.tables["commitments"][cid]
        k = c.get("contract")
        contracts = ctx.world.modules.get("contracts")
        if k and contracts is not None and ctx.state.has("contracts", k):
            contracts.end(ctx, k, status, reason, by=by)

    @staticmethod
    def _vis(c) -> tuple:
        return tuple(sorted({c["payer"], c["payee"]} | ({c["arbiter"]} if c.get("arbiter") else set())))

    # ================================================================== settlement mechanics
    def _set(self, ctx, cid, i, **changes) -> None:
        ctx.emit("commit.milestone", {"id": cid, "milestone": i, "changes": changes},
                 vis=self._vis(ctx.state.tables["commitments"][cid]))

    def _refund_account(self, st, c) -> str:
        bank = self.world.modules["bank"]
        a = st.get("accounts", c["account"])
        if a is not None and a["status"] == "open":
            return c["account"]
        return bank.primary_account(st, c["payer"]) or "acct_fees"   # nobody left to receive it: forfeited

    def _pay_out(self, ctx, cid, i, payee_cents: int, payer_cents: int, fee: int, how: str) -> None:
        st = ctx.state
        c = st.tables["commitments"][cid]
        bank = ctx.world.modules["bank"]
        if fee > 0:
            dst = bank.primary_account(st, c["arbiter"])
            if dst is not None:
                bank.move(ctx, ESCROW, dst, fee, "arbitration", f"commitment {cid} milestone {i}", ref=cid)
            else:                               # an arbiter without an account is not paid; the payer gets it back
                payer_cents += fee
                fee = 0
        if payee_cents > 0:
            dst = bank.primary_account(st, c["payee"])
            if dst is None:                     # the payee no longer has an account: the payer gets it back
                payer_cents += payee_cents
                payee_cents = 0
            else:
                bank.move(ctx, ESCROW, dst, payee_cents, "commitment", f"commitment {cid} milestone {i}", ref=cid)
        if payer_cents > 0:
            bank.move(ctx, ESCROW, self._refund_account(st, c), payer_cents, "refund",
                      f"commitment {cid} milestone {i}", ref=cid)
        ctx.emit("commit.settled", {"id": cid, "milestone": i, "payee_cents": int(payee_cents),
                                    "payer_cents": int(payer_cents), "fee": int(max(0, fee)), "how": how},
                 vis=self._vis(c))
        ctx.unschedule(f"cmt|{cid}|{i}")
        ctx.unschedule(f"rev|{cid}|{i}")
        ctx.unschedule(f"dsp|{cid}|{i}")
        self._maybe_close(ctx, cid)

    def release(self, ctx, cid, i, how="accepted") -> None:
        m = ctx.state.tables["commitments"][cid]["milestones"][i]
        self._set(ctx, cid, i, status="released")
        self._pay_out(ctx, cid, i, m["amount"], 0, 0, how)

    def refund(self, ctx, cid, i, how, breach_by: str | None = None) -> None:
        c = ctx.state.tables["commitments"][cid]
        m = c["milestones"][i]
        self._set(ctx, cid, i, status="refunded")
        if breach_by:
            ctx.emit("commit.breach", {"id": cid, "milestone": i, "by": breach_by}, vis="public")
        self._pay_out(ctx, cid, i, 0, m["amount"], 0, how)

    def resolve(self, ctx, cid, i, payee_share: float, how: str) -> None:
        st = ctx.state
        c = st.tables["commitments"][cid]
        m = c["milestones"][i]
        fee = c["arbiter_fee"] if how == "arbitration" else 0
        rest = m["amount"] - fee
        to_payee = int(round(rest * float(payee_share)))
        self._set(ctx, cid, i, status="resolved", outcome={"payee_share": round(float(payee_share), 4), "how": how})
        self._pay_out(ctx, cid, i, to_payee, rest - to_payee, fee, how)

    def _maybe_close(self, ctx, cid) -> None:
        c = ctx.state.tables["commitments"][cid]
        if c["status"] not in ("active", "disputed"):
            return
        if all(m["status"] in TERMINAL for m in c["milestones"]):
            status = "breached" if c["breaches"] else "completed"
            ctx.emit("commit.status", {"id": cid, "status": status}, vis=self._vis(c))
            self._end_contract(ctx, cid, status, f"commitment {status}", by=c["payee"] if c["breaches"] else None)
        elif any(m["status"] == "disputed" for m in c["milestones"]):
            if c["status"] != "disputed":
                ctx.emit("commit.status", {"id": cid, "status": "disputed"}, vis=self._vis(c))
        elif c["status"] == "disputed":
            ctx.emit("commit.status", {"id": cid, "status": "active"}, vis=self._vis(c))

    def task_met(self, st, c, m) -> bool:
        d = m["deliverable"]
        t = st.get("inst_tasks", d["task"]) if "inst_tasks" in st.tables else None
        if t is None or t["status"] != "done":
            return False
        payee = c["payee"]
        idm = ident(self.world)
        by_payee = t["assignee"] == payee or (payee.startswith("o:") and t["assignee"] is not None
                                              and idm.role(st, payee, t["assignee"]) is not None)
        q = t["quality"]
        return by_payee and (q is None or q >= d["min_quality"])

    def settle_for_wind_up(self, ctx, org: str) -> None:
        """Called before an organisation's accounts close: work delivered to it is paid, open disputes take the
        default split, and milestones it can no longer deliver or pay for return to the payer."""
        st = ctx.state
        for field in ("payer", "payee"):
            for cid in st.find_ids("commitments", field, org):
                c = st.tables["commitments"][cid]
                if c["status"] not in ("active", "disputed"):
                    continue
                for i, m in enumerate(c["milestones"]):
                    c = st.tables["commitments"][cid]
                    m = c["milestones"][i]
                    if m["status"] == "delivered":
                        self.release(ctx, cid, i, "released on wind-up")
                    elif m["status"] == "disputed":
                        self.resolve(ctx, cid, i, c["default_split"], "default")
                    elif m["status"] == "pending":
                        self.refund(ctx, cid, i, "refunded on wind-up")

    # ================================================================== processes
    @process("check")
    def pr_check(self, ctx, commitment, milestone):
        err = contain(ctx, self._check, ctx, commitment, milestone)
        if err is not None:
            ctx.emit("commit.process_failed", {"id": commitment, "error": f"{type(err).__name__}: {err}"[:200]},
                     vis="internal")

    def _check(self, ctx, cid, i):
        from ..goals.metrics import evaluate_predicate
        st = ctx.state
        c = st.get("commitments", cid)
        if c is None or c["milestones"][i]["status"] != "pending":
            return
        m = c["milestones"][i]
        try:
            ok = bool(evaluate_predicate(ctx.world, m["deliverable"]["predicate"]))
        except Exception:  # noqa: BLE001 - a predicate that cannot be evaluated has not been met
            ok = False
        if ok:
            self.release(ctx, cid, i, "predicate met")
        else:
            self.refund(ctx, cid, i, "predicate not met", breach_by=c["payee"])

    @process("due")
    def pr_due(self, ctx, commitment, milestone):
        err = contain(ctx, self._due, ctx, commitment, milestone)
        if err is not None:
            ctx.emit("commit.process_failed", {"id": commitment, "error": f"{type(err).__name__}: {err}"[:200]},
                     vis="internal")

    def _due(self, ctx, cid, i):
        st = ctx.state
        c = st.get("commitments", cid)
        if c is None:
            return
        m = c["milestones"][i]
        if m["status"] != "pending":
            return
        if m["deliverable"]["kind"] == "task" and self.task_met(st, c, m):
            self.release(ctx, cid, i, "task done")
        else:
            self.refund(ctx, cid, i, "not delivered by the due time", breach_by=c["payee"])

    @process("review")
    def pr_review(self, ctx, commitment, milestone):
        err = contain(ctx, self._review_end, ctx, commitment, milestone)
        if err is not None:
            ctx.emit("commit.process_failed", {"id": commitment, "error": f"{type(err).__name__}: {err}"[:200]},
                     vis="internal")

    def _review_end(self, ctx, cid, i):
        c = ctx.state.get("commitments", cid)
        if c is not None and c["milestones"][i]["status"] == "delivered":
            self.release(ctx, cid, i, "accepted by silence")

    @process("dispute_end")
    def pr_dispute_end(self, ctx, commitment, milestone):
        err = contain(ctx, self._dispute_end, ctx, commitment, milestone)
        if err is not None:
            ctx.emit("commit.process_failed", {"id": commitment, "error": f"{type(err).__name__}: {err}"[:200]},
                     vis="internal")

    def _dispute_end(self, ctx, cid, i):
        c = ctx.state.get("commitments", cid)
        if c is not None and c["milestones"][i]["status"] == "disputed":
            self.resolve(ctx, cid, i, c["default_split"], "default")

    @process("expire")
    def pr_expire(self, ctx):
        for oid in ctx.state.find_ids("commit_offers", "status", "open"):
            if ctx.state.tables["commit_offers"][oid]["expires"] <= ctx.now:
                o = ctx.state.tables["commit_offers"][oid]
                ctx.emit("commit.offer_status", {"id": oid, "status": "expired"}, vis=(o["from"], o["to"]))

    # ================================================================== endpoints: negotiation
    @endpoint("offer", S.obj({
        "to": S.s_str("counterparty principal or handle", maxLength=60),
        "terms": S.s_any_obj("{title, payer: $me|$you, milestones: [{amount_cents, deliverable: {kind: acceptance} | "
                             "{kind: predicate, predicate} | {kind: task, task, min_quality}, due_in_s}], arbiter?, "
                             "arbiter_fee_cents?, review_s?, dispute_s?, default_split?}"),
        "message": S.s_str(maxLength=4000), "as_org": S.s_str(maxLength=40),
        "account": S.s_str("your account for the escrow deposit if you pay", maxLength=40),
        "expires_in_s": S.s_int(minimum=3600, maximum=30 * DAY)}, ["to", "terms"]), cost_s=60)
    def ep_offer(self, ctx, to, terms, message="", as_org=None, account=None, expires_in_s=3 * DAY):
        """Offer a commitment. The terms are validated now; nothing binds until the counterparty accepts, and the
        payer's money goes into escrow at that moment."""
        st = ctx.state
        me = self.party(ctx, as_org)
        you = ident(ctx.world).resolve_handle(st, to)
        if you == me:
            raise InvalidAction("cannot commit to yourself")
        return {"offer": self.offer(ctx, me, you, terms, message, account, expires_in_s)}

    @endpoint("respond", S.obj({
        "offer": S.s_str(maxLength=40), "action": S.s_enum(["accept", "reject", "counter"]),
        "terms": S.s_any_obj("for a counter: your terms ($me = you now)"), "message": S.s_str(maxLength=4000),
        "account": S.s_str("your account for the escrow deposit if you pay", maxLength=40),
        "as_org": S.s_str(maxLength=40)}, ["offer", "action"]), cost_s=60)
    def ep_respond(self, ctx, offer, action, terms=None, message="", account=None, as_org=None):
        """Accept (the commitment starts and the deposit moves to escrow), reject, or counter with other terms."""
        st = ctx.state
        o = st.get("commit_offers", offer)
        if o is None or o["status"] != "open":
            raise NotFound("no open offer")
        who = self.party(ctx, as_org)
        if who != o["to"]:
            raise Forbidden("not addressed to you")
        if action == "reject":
            ctx.emit("commit.offer_status", {"id": offer, "status": "rejected", "message": str(message)[:1000]},
                     vis=(o["from"], o["to"]))
            return {"status": "rejected"}
        if action == "counter":
            if not terms:
                raise InvalidAction("a counter needs terms")
            ctx.emit("commit.offer_status", {"id": offer, "status": "countered"}, vis=(o["from"], o["to"]))
            return {"status": "countered", "offer": self.offer(ctx, who, o["from"], terms, message, account,
                                                               3 * DAY, offer)}
        return {"status": "accepted", "commitment": self.accept(ctx, offer, who, account)}

    @endpoint("withdraw", S.obj({"offer": S.s_str(maxLength=40), "as_org": S.s_str(maxLength=40)}, ["offer"]))
    def ep_withdraw(self, ctx, offer, as_org=None):
        """Withdraw an open offer you made."""
        o = ctx.state.get("commit_offers", offer)
        if o is None or o["status"] != "open" or o["from"] != self.party(ctx, as_org):
            raise NotFound("no open offer of yours")
        ctx.emit("commit.offer_status", {"id": offer, "status": "withdrawn"}, vis=(o["from"], o["to"]))
        return {"ok": True}

    @endpoint("inbox", S.obj({"as_org": S.s_str(maxLength=40), "status": S.s_enum(["open", "any"])}),
              read_only=True)
    def ep_inbox(self, ctx, as_org=None, status="open"):
        """Offers you (or your organisation) received and made."""
        st = ctx.state
        me = self.party(ctx, as_org)
        out = []
        for side in ("to", "from"):
            for oid in st.find_ids("commit_offers", side, me):
                o = st.tables["commit_offers"][oid]
                if status == "open" and o["status"] != "open":
                    continue
                out.append({"offer": oid, "direction": "received" if side == "to" else "sent",
                            **{k: o[k] for k in ("from", "to", "terms", "message", "status", "round", "thread",
                                                 "expires", "commitment")}})
        return out[-50:]

    @endpoint("thread", S.obj({"offer": S.s_str(maxLength=40)}, ["offer"]), read_only=True)
    def ep_thread(self, ctx, offer):
        """The negotiation history of an offer's thread (its two parties only)."""
        st = ctx.state
        o = st.get("commit_offers", offer)
        if o is None:
            raise NotFound("no offer")
        if not (ctx.world.principals_of(ctx.actor) & {o["from"], o["to"]}):
            raise Forbidden("not your negotiation")
        return [{"offer": k, **{x: st.tables["commit_offers"][k][x] for x in ("from", "to", "terms", "message",
                                                                              "status", "round", "t")}}
                for k in st.find_ids("commit_offers", "thread", o["thread"])]

    # ================================================================== endpoints: delivery, disputes, resolution
    def _mine(self, ctx, commitment, as_org, role: str):
        st = ctx.state
        c = st.get("commitments", commitment)
        if c is None:
            raise NotFound("no commitment")
        who = self.party(ctx, as_org)
        if role == "party":
            if who not in (c["payer"], c["payee"]):
                raise Forbidden("not a party")
        elif c[role] != who:
            raise Forbidden(f"only the {role} can do this")
        return c, who

    def _milestone(self, c, i) -> dict:
        if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(c["milestones"]):
            raise NotFound("no such milestone")
        return c["milestones"][i]

    @endpoint("deliver", S.obj({"commitment": S.s_str(maxLength=40), "milestone": S.s_int(minimum=0, maximum=7),
                                "note": S.s_str(maxLength=2000), "ref": S.s_str("what was delivered (a repository, "
                                                                                "artifact or document id)",
                                                                                maxLength=120),
                                "as_org": S.s_str(maxLength=40)}, ["commitment", "milestone"]), cost_s=30)
    def ep_deliver(self, ctx, commitment, milestone, note="", ref="", as_org=None):
        """The payee reports a milestone delivered. For an acceptance deliverable the payer then accepts or
        disputes within the review window (silence releases the money); a task deliverable is checked now."""
        st = ctx.state
        c, _ = self._mine(ctx, commitment, as_org, "payee")
        m = self._milestone(c, milestone)
        if m["status"] != "pending":
            raise Conflict(f"milestone is {m['status']}")
        kind = m["deliverable"]["kind"]
        if kind == "predicate":
            raise InvalidAction("a predicate deliverable is checked by the world at its due time")
        if kind == "task":
            if not self.task_met(st, c, m):
                raise Conflict("the task is not done by you at the agreed quality")
            self.release(ctx, commitment, milestone, "task done")
            return {"status": "released"}
        self._set(ctx, commitment, milestone, status="delivered", delivered_t=ctx.now, note=f"{note[:2000]} {ref}"[:2100])
        ctx.unschedule(f"cmt|{commitment}|{milestone}")
        ctx.schedule(ctx.now + c["review_s"], "commit.review", {"commitment": commitment, "milestone": milestone},
                     prio=37, key=f"rev|{commitment}|{milestone}")
        return {"status": "delivered", "review_until": ctx.now + c["review_s"]}

    @endpoint("accept", S.obj({"commitment": S.s_str(maxLength=40), "milestone": S.s_int(minimum=0, maximum=7),
                               "as_org": S.s_str(maxLength=40)}, ["commitment", "milestone"]))
    def ep_accept(self, ctx, commitment, milestone, as_org=None):
        """The payer accepts a delivered milestone: its money leaves escrow for the payee."""
        c, _ = self._mine(ctx, commitment, as_org, "payer")
        m = self._milestone(c, milestone)
        if m["status"] != "delivered":
            raise Conflict("nothing delivered to accept")
        self.release(ctx, commitment, milestone, "accepted")
        return {"status": "released"}

    @endpoint("dispute", S.obj({"commitment": S.s_str(maxLength=40), "milestone": S.s_int(minimum=0, maximum=7),
                                "reason": S.s_str(maxLength=2000), "as_org": S.s_str(maxLength=40)},
                               ["commitment", "milestone", "reason"]))
    def ep_dispute(self, ctx, commitment, milestone, reason, as_org=None):
        """The payer disputes a delivered milestone within the review window: its money stays in escrow until
        the parties settle, the arbiter rules, or the dispute window closes (default split)."""
        c, who = self._mine(ctx, commitment, as_org, "payer")
        m = self._milestone(c, milestone)
        if m["status"] != "delivered":
            raise Conflict("only a delivered milestone can be disputed")
        deadline = ctx.now + c["dispute_s"]
        self._set(ctx, commitment, milestone, status="disputed",
                  dispute={"by": who, "reason": reason[:2000], "t": ctx.now, "deadline": deadline})
        ctx.unschedule(f"rev|{commitment}|{milestone}")
        ctx.schedule(deadline, "commit.dispute_end", {"commitment": commitment, "milestone": milestone}, prio=38,
                     key=f"dsp|{commitment}|{milestone}")
        self._maybe_close(ctx, commitment)
        hook = ctx.world.hooks.get("on_commit_dispute")
        if hook:
            hook(ctx, commitment, milestone)
        return {"status": "disputed", "resolves_by": deadline}

    @endpoint("settle", S.obj({"commitment": S.s_str(maxLength=40), "milestone": S.s_int(minimum=0, maximum=7),
                               "payee_share": S.s_num("share of the milestone for the payee", minimum=0.0,
                                                      maximum=1.0),
                               "as_org": S.s_str(maxLength=40)}, ["commitment", "milestone", "payee_share"]))
    def ep_settle(self, ctx, commitment, milestone, payee_share, as_org=None):
        """Propose a settlement of a disputed milestone; it takes effect when both parties propose the same split."""
        c, who = self._mine(ctx, commitment, as_org, "party")
        m = self._milestone(c, milestone)
        if m["status"] != "disputed":
            raise Conflict("not in dispute")
        share = round(float(payee_share), 4)
        props = dict(m["proposals"], **{who: share})
        self._set(ctx, commitment, milestone, proposals=props)
        other = c["payee"] if who == c["payer"] else c["payer"]
        if props.get(other) == share:
            self.resolve(ctx, commitment, milestone, share, "settlement")
            return {"status": "resolved", "payee_share": share}
        return {"status": "proposed", "payee_share": share}

    @endpoint("rule", S.obj({"commitment": S.s_str(maxLength=40), "milestone": S.s_int(minimum=0, maximum=7),
                             "payee_share": S.s_num(minimum=0.0, maximum=1.0), "reason": S.s_str(maxLength=2000),
                             "as_org": S.s_str(maxLength=40)}, ["commitment", "milestone", "payee_share"]))
    def ep_rule(self, ctx, commitment, milestone, payee_share, reason="", as_org=None):
        """The arbiter named in the terms rules on a disputed milestone (and is paid its fee from it)."""
        c, _ = self._mine(ctx, commitment, as_org, "arbiter")
        m = self._milestone(c, milestone)
        if m["status"] != "disputed":
            raise Conflict("not in dispute")
        self.resolve(ctx, commitment, milestone, float(payee_share), "arbitration")
        return {"status": "resolved", "payee_share": round(float(payee_share), 4)}

    @endpoint("cancel", S.obj({"commitment": S.s_str(maxLength=40), "as_org": S.s_str(maxLength=40)},
                              ["commitment"]))
    def ep_cancel(self, ctx, commitment, as_org=None):
        """Request mutual cancellation: once both parties ask, every milestone not yet delivered is refunded."""
        c, who = self._mine(ctx, commitment, as_org, "party")
        if c["status"] != "active":
            raise Conflict("only an active commitment can be cancelled")
        req = sorted(set(c["cancel_requests"]) | {who})
        ctx.emit("commit.cancel_request", {"id": commitment, "requests": req}, vis=self._vis(c))
        if set(req) == {c["payer"], c["payee"]}:
            for i, m in enumerate(ctx.state.tables["commitments"][commitment]["milestones"]):
                if m["status"] == "pending":
                    self.refund(ctx, commitment, i, "cancelled by both parties")
            return {"status": ctx.state.tables["commitments"][commitment]["status"]}
        return {"status": "cancellation requested"}

    @endpoint("view", S.obj({"commitment": S.s_str(maxLength=40)}, ["commitment"]), read_only=True)
    def ep_view(self, ctx, commitment):
        """A commitment's terms and state (its parties and arbiter only)."""
        c = ctx.state.get("commitments", commitment)
        if c is None:
            raise NotFound("no commitment")
        if not (ctx.world.principals_of(ctx.actor) & set(self._vis(c))):
            raise Forbidden("not a party")
        return dict(c, id=commitment)

    @endpoint("mine", S.obj({"as_org": S.s_str(maxLength=40)}), read_only=True)
    def ep_mine(self, ctx, as_org=None):
        """Commitments where you are payer, payee or arbiter."""
        st = ctx.state
        me = self.party(ctx, as_org)
        out = []
        for f in ("payer", "payee", "arbiter"):
            for cid in st.find_ids("commitments", f, me):
                c = st.tables["commitments"][cid]
                out.append({"commitment": cid, "role": f, "title": c["title"], "status": c["status"],
                            "milestones": [{"amount": m["amount"], "status": m["status"], "due": m["due"],
                                            "kind": m["deliverable"]["kind"]} for m in c["milestones"]]})
        return out[-50:]

    def record(self, world, who: str) -> dict:
        st = world.state
        out = {"principal": who, "completed": 0, "breaches": 0, "active": 0, "disputes_raised": 0,
               "disputes_against": 0, "settled": 0, "arbitrated": 0, "defaulted": 0, "arbitrations_served": 0}
        for f in ("payer", "payee"):
            for cid in st.find_ids("commitments", f, who):
                c = st.tables["commitments"][cid]
                out["completed"] += c["status"] == "completed"
                out["active"] += c["status"] in ("active", "disputed")
                out["breaches"] += sum(1 for b in c["breaches"] if b["by"] == who)
                for m in c["milestones"]:
                    if m["dispute"] is None:
                        continue
                    if m["dispute"]["by"] == who:
                        out["disputes_raised"] += 1
                    else:
                        out["disputes_against"] += 1
                    how = (m["outcome"] or {}).get("how")
                    out["settled"] += how == "settlement"
                    out["arbitrated"] += how == "arbitration"
                    out["defaulted"] += how == "default"
        out["arbitrations_served"] = sum(1 for cid in st.find_ids("commitments", "arbiter", who)
                                         for m in st.tables["commitments"][cid]["milestones"]
                                         if (m["outcome"] or {}).get("how") == "arbitration")
        return out

    @endpoint("record", S.obj({"principal": S.s_str(maxLength=60)}, ["principal"]), read_only=True)
    def ep_record(self, ctx, principal):
        """Public track record of a principal's commitments: completed, breaches, disputes and how they ended
        (counts only)."""
        return self.record(ctx.world, ident(ctx.world).resolve_handle(ctx.state, principal))

    # ================================================================== appliers
    @applier("commit.offer")
    def ap_offer(self, ev):
        p = ev.payload
        self.world.state.insert("commit_offers", p["id"], {
            "from": p["from"], "to": p["to"], "terms": p["terms"], "concrete": p["concrete"], "message": p["message"],
            "account": p["account"], "expires": p["expires"], "thread": p["thread"], "round": p["round"],
            "parent": p["parent"], "status": "open", "commitment": None, "t": ev.t})

    @applier("commit.offer_status")
    def ap_offer_status(self, ev):
        p = ev.payload
        ch = {"status": p["status"]}
        if p.get("commitment"):
            ch["commitment"] = p["commitment"]
        self.world.state.patch("commit_offers", p["id"], **ch)

    @applier("commit.created")
    def ap_created(self, ev):
        p = ev.payload
        self.world.state.insert("commitments", p["id"], {
            "offer": p["offer"], "thread": p["thread"], "payer": p["payer"], "payee": p["payee"],
            "arbiter": p["arbiter"], "arbiter_fee": p["arbiter_fee"], "title": p["title"],
            "milestones": p["milestones"], "account": p["account"], "review_s": p["review_s"],
            "dispute_s": p["dispute_s"], "default_split": p["default_split"], "held": p["held"],
            "status": "active", "breaches": [], "cancel_requests": [], "t": ev.t})

    @applier("commit.contract")
    def ap_contract(self, ev):
        self.world.state.patch("commitments", ev.payload["id"], contract=ev.payload["contract"])

    @applier("commit.milestone")
    def ap_milestone(self, ev):
        st = self.world.state
        p = ev.payload
        c = st.tables["commitments"][p["id"]]
        ms = list(c["milestones"])
        ms[p["milestone"]] = {**ms[p["milestone"]], **p["changes"]}
        st.patch("commitments", p["id"], milestones=ms)

    @applier("commit.settled")
    def ap_settled(self, ev):
        st = self.world.state
        p = ev.payload
        c = st.tables["commitments"][p["id"]]
        st.patch("commitments", p["id"], held=c["held"] - p["payee_cents"] - p["payer_cents"] - p["fee"])

    @applier("commit.breach")
    def ap_breach(self, ev):
        st = self.world.state
        p = ev.payload
        c = st.tables["commitments"][p["id"]]
        st.patch("commitments", p["id"], breaches=list(c["breaches"]) + [{"milestone": p["milestone"], "by": p["by"],
                                                                          "t": ev.t}])

    @applier("commit.status")
    def ap_status(self, ev):
        self.world.state.patch("commitments", ev.payload["id"], status=ev.payload["status"], updated=ev.t)

    @applier("commit.cancel_request")
    def ap_cancel_request(self, ev):
        self.world.state.patch("commitments", ev.payload["id"], cancel_requests=ev.payload["requests"])

    @applier("commit.process_failed")
    def ap_process_failed(self, ev):
        return None

    # ================================================================== invariants
    def check_invariants(self, world):
        st = world.state
        errs = []
        held = 0
        for cid, c in st.tables["commitments"].items():
            open_amt = sum(m["amount"] for m in c["milestones"] if m["status"] not in TERMINAL)
            if open_amt != c["held"]:
                errs.append(f"{cid}: holds {c['held']} for {open_amt} in open milestones")
            held += c["held"]
        bal = st.tables["accounts"][ESCROW]["balance"] if st.has("accounts", ESCROW) else 0
        if bal != held:
            errs.append(f"commitment escrow {bal} != held {held}")
        return errs

    def observe(self, world, actor):
        st = world.state
        n = sum(1 for p in world.principals_of(actor) for o in st.find("commit_offers", "to", p)
                if o["status"] == "open")
        d = sum(1 for p in world.principals_of(actor) for f in ("payer", "payee", "arbiter")
                for c in st.find("commitments", f, p) if c["status"] == "disputed")
        return {"offers_open": n, "disputes": d} if (n or d) else None
