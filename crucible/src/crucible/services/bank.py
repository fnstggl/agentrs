"""Payments and banking on a double-entry ledger (integer cents).

Every movement of money is one `bank.txn` event whose postings sum to zero, so the sum of all balances is
always exactly 0. Money enters the private economy only from system accounts with a declared role:

* `acct_genesis`   - initial endowments written by the world generator,
* `acct_households`- the household sector: receives exogenous income each period and spends it on services,
* `acct_external`  - the rest of the world: sells hardware/energy, pays exogenous wages, collects fees.

System accounts may run negative (they are the sources); every other account must stay >= -overdraft.
Invariants: postings balance; non-system accounts respect their overdraft; balances equal the sum of postings.
"""
from __future__ import annotations

from ..core import schema as S
from ..core.errors import Conflict, Forbidden, InsufficientFunds, InvalidAction, NotFound
from ..core.kernel import SYSTEM, Ctx, Module, applier, endpoint, process, register_module
from .identity import ident

SYSTEM_ACCOUNTS = {"acct_genesis": "genesis", "acct_households": "households", "acct_external": "external",
                   "acct_fees": "fees"}
MAX_AMOUNT = 10**12


def fmt(cents: int) -> str:
    s = "-" if cents < 0 else ""
    c = abs(int(cents))
    return f"{s}${c // 100:,}.{c % 100:02d}"


@register_module
class Bank(Module):
    name = "bank"
    depends = ("identity",)

    def install(self, world):
        st = world.state
        st.declare_table("accounts", index=("owner",))
        st.declare_table("txns", index=("kind",))
        st.declare_table("acct_txns", index=("acct",))
        st.declare_table("invoices", index=("payer", "payee_owner", "status"))
        st.declare_table("mandates", index=("payer_acct", "payee"))
        world.hooks["bank"] = self

    def genesis(self, ctx):
        for aid, role in SYSTEM_ACCOUNTS.items():
            # households is a budget-constrained pool (funded by exogenous income), not an unlimited source
            kind = "pool" if role == "households" else "system"
            ctx.emit("bank.account", {"id": aid, "owner": SYSTEM, "kind": kind, "name": role, "overdraft": 0},
                     vis="internal")

    # ------------------------------------------------------------------ internal API
    def open_account(self, ctx: Ctx, owner: str, name: str, kind: str = "checking", overdraft: int = 0) -> str:
        aid = ctx.state.new_id("acct")
        ctx.emit("bank.account", {"id": aid, "owner": owner, "kind": kind, "name": name, "overdraft": overdraft},
                 vis=(owner,))
        return aid

    def primary_account(self, state, owner: str) -> str | None:
        ids = [a for a in state.find_ids("accounts", "owner", owner)
               if state.tables["accounts"][a]["status"] == "open" and state.tables["accounts"][a]["kind"] == "checking"]
        return ids[0] if ids else None

    def balance(self, state, aid: str) -> int:
        return state.tables["accounts"][aid]["balance"]

    def available(self, state, aid: str) -> int:
        a = state.tables["accounts"][aid]
        return a["balance"] + a["overdraft"] if a["kind"] != "system" else MAX_AMOUNT

    def post(self, ctx: Ctx, postings: list[tuple[str, int]], kind: str, memo: str = "", ref: str | None = None,
             vis_extra: tuple = ()) -> str:
        """Apply a balanced set of postings atomically. Positive = credit (money in). No authorisation check:
        callers authorise first. Raises InsufficientFunds if a non-system account would exceed its overdraft."""
        if not postings:
            raise InvalidAction("empty transaction")
        if sum(a for _, a in postings) != 0:
            raise InvalidAction("unbalanced transaction")
        st = ctx.state
        net: dict[str, int] = {}
        for acct, amt in postings:
            if not isinstance(amt, int) or abs(amt) > MAX_AMOUNT:
                raise InvalidAction("amounts must be integer cents")
            a = st.get("accounts", acct)
            if a is None:
                raise NotFound(f"no account {acct}")
            if a["status"] != "open":
                raise Forbidden(f"account {acct} is {a['status']}")
            net[acct] = net.get(acct, 0) + amt
        for acct, d in net.items():
            if d < 0 and self.available(st, acct) + d < 0:
                raise InsufficientFunds(f"{acct} has {fmt(self.available(st, acct))} available, needs {fmt(-d)}",
                                        account=acct)
        tid = st.new_id("tx")
        owners = {st.tables["accounts"][a]["owner"] for a in net} - {SYSTEM}
        ctx.emit("bank.txn", {"id": tid, "postings": [[a, d] for a, d in sorted(net.items()) if d != 0], "kind": kind,
                              "memo": memo[:300], "ref": ref, "by": ctx.actor},
                 vis=tuple(sorted(owners | set(vis_extra))))
        return tid

    def move(self, ctx: Ctx, src: str, dst: str, amount: int, kind: str, memo: str = "", ref: str | None = None) -> str:
        if amount <= 0:
            raise InvalidAction("amount must be positive")
        if src == dst:
            raise InvalidAction("source and destination are the same account")
        return self.post(ctx, [(src, -amount), (dst, amount)], kind, memo, ref)

    def account_owner(self, state, aid: str) -> str:
        a = state.get("accounts", aid)
        if a is None:
            raise NotFound(f"no account {aid}")
        return a["owner"]

    def resolve_account(self, state, x: str) -> str:
        """Accept an account id or a principal/handle (→ its primary checking account)."""
        if x.startswith("acct_"):
            if not state.has("accounts", x):
                raise NotFound(f"no account {x}")
            return x
        pid = ident(self.world).resolve_handle(state, x)
        aid = self.primary_account(state, pid)
        if aid is None:
            raise NotFound(f"{pid} has no open account")
        return aid

    # ------------------------------------------------------------------ endpoints
    @endpoint("open_account", S.obj({"name": S.s_str(maxLength=80), "owner": S.s_str("org to open for", maxLength=40)}),
              cost_s=30)
    def ep_open(self, ctx, name="checking", owner=None):
        """Open a checking account for yourself or an organisation you administer (companies must be registered)."""
        owner = ident(ctx.world).resolve_handle(ctx.state, owner) if owner else ctx.actor
        if owner != ctx.actor:
            ident(ctx.world).authorize(ctx, owner, "org.admin")
        if owner.startswith("o:"):
            reg = ctx.world.hooks.get("is_registered_company")
            if reg and not reg(ctx.state, owner):
                raise Forbidden("organisations need a registered company (biz.register) to open accounts")
        if len(ctx.state.find_ids("accounts", "owner", owner)) >= 5:
            raise Conflict("account limit reached")
        return {"account": self.open_account(ctx, owner, name)}

    @endpoint("accounts", S.obj(), read_only=True)
    def ep_accounts(self, ctx):
        """Accounts you can see: your own and those of organisations you administer or hold bank grants for."""
        out = []
        for owner in sorted(ctx.world.principals_of(ctx.actor) | self._granted_owners(ctx)):
            for aid in ctx.state.find_ids("accounts", "owner", owner):
                a = ctx.state.tables["accounts"][aid]
                out.append({"account": aid, "owner": owner, "name": a["name"], "balance": a["balance"],
                            "balance_fmt": fmt(a["balance"]), "overdraft": a["overdraft"], "status": a["status"]})
        return out

    def _granted_owners(self, ctx) -> set[str]:
        return {g["owner"] for g in ctx.state.find("grants", "grantee", ctx.actor)
                if g["status"] == "active" and g["cap"].startswith("bank")}

    @endpoint("balance", S.obj({"account": S.s_str(maxLength=40)}, ["account"]), read_only=True)
    def ep_balance(self, ctx, account):
        """Balance and status of one account you may read (bank.read)."""
        owner = self.account_owner(ctx.state, account)
        ident(ctx.world).authorize(ctx, owner, "bank.read")
        a = ctx.state.tables["accounts"][account]
        return {"account": account, "balance": a["balance"], "balance_fmt": fmt(a["balance"]), "status": a["status"]}

    @endpoint("transfer", S.obj({"from_account": S.s_str(maxLength=40),
                                 "to": S.s_str("account id or principal/handle (its primary account)", maxLength=60),
                                 "amount": S.s_int("cents", minimum=1, maximum=MAX_AMOUNT),
                                 "memo": S.s_str(maxLength=300), "ref": S.s_str(maxLength=60)},
                                ["from_account", "to", "amount"]), cost_s=10)
    def ep_transfer(self, ctx, from_account, to, amount, memo="", ref=None):
        """Send money. Requires bank.spend on the source account's owner (own accounts need nothing)."""
        owner = self.account_owner(ctx.state, from_account)
        ident(ctx.world).authorize(ctx, owner, "bank.spend", amount=amount, scope=from_account)
        dst = self.resolve_account(ctx.state, to)
        if ctx.state.tables["accounts"][dst]["owner"] == SYSTEM:
            # system accounts, the household pool and escrow accounts only move through their own mechanisms
            raise Forbidden("cannot pay system-operated accounts directly")
        tid = self.move(ctx, from_account, dst, amount, "transfer", memo, ref)
        return {"txn": tid, "balance": ctx.state.tables["accounts"][from_account]["balance"]}

    @endpoint("statement", S.obj({"account": S.s_str(maxLength=40), "limit": S.s_int(minimum=1, maximum=500)},
                                 ["account"]), read_only=True)
    def ep_statement(self, ctx, account, limit=50):
        """Recent ledger entries of an account you may read, newest first."""
        owner = self.account_owner(ctx.state, account)
        ident(ctx.world).authorize(ctx, owner, "bank.read")
        ids = ctx.state.find_ids("acct_txns", "acct", account)[-limit:]
        out = []
        for lid in reversed(ids):
            e = ctx.state.tables["acct_txns"][lid]
            tx = ctx.state.tables["txns"][e["txn"]]
            out.append({"txn": e["txn"], "t": tx["t"], "amount": e["amount"], "kind": tx["kind"], "memo": tx["memo"],
                        "counterparties": [a for a, _ in tx["postings"] if a != account], "ref": tx["ref"]})
        return out

    @endpoint("invoice", S.obj({"payer": S.s_str("principal or handle", maxLength=60),
                                "to_account": S.s_str(maxLength=40), "amount": S.s_int(minimum=1, maximum=MAX_AMOUNT),
                                "memo": S.s_str(maxLength=300), "due_in": S.s_int("seconds", minimum=0),
                                "ref": S.s_str(maxLength=60)},
                               ["payer", "to_account", "amount"]), cost_s=10)
    def ep_invoice(self, ctx, payer, to_account, amount, memo="", due_in=7 * 86400, ref=None):
        """Request a payment. The payer sees it and may pay it; nothing moves until they do."""
        payee_owner = self.account_owner(ctx.state, to_account)
        ident(ctx.world).authorize(ctx, payee_owner, "bank.read")
        payer = ident(ctx.world).resolve_handle(ctx.state, payer)
        iid = ctx.state.new_id("inv")
        ctx.emit("bank.invoice", {"id": iid, "payer": payer, "payee_acct": to_account, "payee_owner": payee_owner,
                                  "amount": amount, "memo": memo[:300], "due": ctx.now + due_in, "ref": ref},
                 vis=(payer, payee_owner))
        return {"invoice": iid}

    @endpoint("invoices", S.obj({"role": S.s_enum(["payer", "payee"])}), read_only=True)
    def ep_invoices(self, ctx, role="payer"):
        """Invoices addressed to you (role=payer) or issued by you (role=payee)."""
        field = "payer" if role == "payer" else "payee_owner"
        out = []
        for owner in sorted(ctx.world.principals_of(ctx.actor)):
            for iid in ctx.state.find_ids("invoices", field, owner):
                out.append(dict(ctx.state.tables["invoices"][iid], id=iid))
        return out

    @endpoint("pay_invoice", S.obj({"invoice": S.s_str(maxLength=40), "from_account": S.s_str(maxLength=40)},
                                   ["invoice", "from_account"]), cost_s=10)
    def ep_pay_invoice(self, ctx, invoice, from_account):
        """Pay an open invoice addressed to you from one of your accounts."""
        inv = ctx.state.get("invoices", invoice)
        if inv is None or inv["status"] != "open":
            raise NotFound("no open invoice")
        owner = self.account_owner(ctx.state, from_account)
        if owner != inv["payer"]:
            raise Forbidden("pay from an account of the invoiced principal")
        ident(ctx.world).authorize(ctx, owner, "bank.spend", amount=inv["amount"], scope=from_account)
        tid = self.move(ctx, from_account, inv["payee_acct"], inv["amount"], "invoice", inv["memo"], invoice)
        ctx.emit("bank.invoice_paid", {"id": invoice, "txn": tid}, vis=(inv["payer"], inv["payee_owner"]))
        return {"txn": tid}

    @endpoint("authorize_debit", S.obj({"from_account": S.s_str(maxLength=40),
                                        "payee": S.s_str("principal allowed to pull funds", maxLength=60),
                                        "max_per_period": S.s_int(minimum=1, maximum=MAX_AMOUNT),
                                        "period_s": S.s_int(minimum=3600), "purpose": S.s_str(maxLength=120)},
                                       ["from_account", "payee", "max_per_period"]), cost_s=10)
    def ep_mandate(self, ctx, from_account, payee, max_per_period, period_s=30 * 86400, purpose=""):
        """Direct-debit mandate: lets a payee (e.g. a cloud provider or employee contract) pull up to a limit."""
        owner = self.account_owner(ctx.state, from_account)
        ident(ctx.world).authorize(ctx, owner, "bank.spend", scope=from_account)
        payee = ident(ctx.world).resolve_handle(ctx.state, payee)
        mid = ctx.state.new_id("mandate")
        ctx.emit("bank.mandate", {"id": mid, "payer_acct": from_account, "payer": owner, "payee": payee,
                                  "max": max_per_period, "period": period_s, "purpose": purpose[:120]},
                 vis=(owner, payee))
        return {"mandate": mid}

    @endpoint("cancel_mandate", S.obj({"mandate": S.s_str(maxLength=40)}, ["mandate"]))
    def ep_cancel_mandate(self, ctx, mandate):
        """Cancel a direct-debit mandate you granted."""
        m = ctx.state.get("mandates", mandate)
        if m is None:
            raise NotFound("no mandate")
        ident(ctx.world).authorize(ctx, m["payer"], "bank.spend")
        ctx.emit("bank.mandate_status", {"id": mandate, "status": "cancelled"}, vis=(m["payer"], m["payee"]))
        return {"ok": True}

    def debit(self, ctx: Ctx, mandate: str, dst: str, amount: int, kind: str, memo: str = "", ref=None) -> str:
        """Pull money under a mandate (used by other modules on behalf of the payee)."""
        m = ctx.state.get("mandates", mandate)
        if m is None or m["status"] != "active":
            raise Forbidden("mandate not active")
        used = m["used"] if ctx.now - m["window_start"] < m["period"] else 0
        if used + amount > m["max"]:
            raise Forbidden("mandate limit exceeded")
        tid = self.move(ctx, m["payer_acct"], dst, amount, kind, memo, ref)
        ctx.emit("bank.mandate_used", {"id": mandate, "amount": amount}, vis="internal")
        return tid

    # ------------------------------------------------------------------ appliers
    @applier("bank.account")
    def ap_account(self, ev):
        p = ev.payload
        self.world.state.insert("accounts", p["id"], {"owner": p["owner"], "kind": p["kind"], "name": p["name"],
                                                     "balance": 0, "overdraft": p["overdraft"], "status": "open",
                                                     "created": ev.t})

    @applier("bank.txn")
    def ap_txn(self, ev):
        st = self.world.state
        p = ev.payload
        st.insert("txns", p["id"], {"t": ev.t, "postings": p["postings"], "kind": p["kind"], "memo": p["memo"],
                                    "ref": p["ref"], "by": p["by"]})
        for acct, d in p["postings"]:
            a = st.tables["accounts"][acct]
            st.patch("accounts", acct, balance=a["balance"] + d)
            st.insert("acct_txns", f"{p['id']}|{acct}", {"acct": acct, "txn": p["id"], "amount": d, "t": ev.t})
        for fn in self.world.hooks.get("txn_observers", ()):   # derived per-owner accounting (ops module)
            fn(ev)

    @applier("bank.account_status")
    def ap_status(self, ev):
        self.world.state.patch("accounts", ev.payload["id"], status=ev.payload["status"])

    @applier("bank.overdraft")
    def ap_overdraft(self, ev):
        self.world.state.patch("accounts", ev.payload["id"], overdraft=ev.payload["overdraft"])

    @applier("bank.invoice")
    def ap_invoice(self, ev):
        p = ev.payload
        self.world.state.insert("invoices", p["id"], {"payer": p["payer"], "payee_acct": p["payee_acct"],
                                                     "payee_owner": p["payee_owner"], "amount": p["amount"],
                                                     "memo": p["memo"], "due": p["due"], "ref": p["ref"],
                                                     "status": "open", "t": ev.t})

    @applier("bank.invoice_paid")
    def ap_invoice_paid(self, ev):
        self.world.state.patch("invoices", ev.payload["id"], status="paid", txn=ev.payload["txn"])

    @applier("bank.invoice_status")
    def ap_invoice_status(self, ev):
        self.world.state.patch("invoices", ev.payload["id"], status=ev.payload["status"])

    @applier("bank.mandate")
    def ap_mandate(self, ev):
        p = ev.payload
        self.world.state.insert("mandates", p["id"], {"payer_acct": p["payer_acct"], "payer": p["payer"],
                                                     "payee": p["payee"], "max": p["max"], "period": p["period"],
                                                     "purpose": p["purpose"], "status": "active", "used": 0,
                                                     "window_start": ev.t})

    @applier("bank.mandate_used")
    def ap_mandate_used(self, ev):
        st = self.world.state
        m = st.tables["mandates"][ev.payload["id"]]
        if ev.t - m["window_start"] >= m["period"]:
            st.patch("mandates", ev.payload["id"], used=ev.payload["amount"], window_start=ev.t)
        else:
            st.patch("mandates", ev.payload["id"], used=m["used"] + ev.payload["amount"])

    @applier("bank.mandate_status")
    def ap_mandate_status(self, ev):
        self.world.state.patch("mandates", ev.payload["id"], status=ev.payload["status"])

    # ------------------------------------------------------------------ invariants & observation
    def check_invariants(self, world):
        st = world.state
        errs = []
        total = 0
        for aid, a in st.tables["accounts"].items():
            total += a["balance"]
            if a["kind"] != "system" and a["balance"] < -a["overdraft"]:
                errs.append(f"{aid} below overdraft: {a['balance']} < -{a['overdraft']}")
        if total != 0:
            errs.append(f"money not conserved: sum of balances = {total}")
        return errs

    def check_ledger_consistency(self, world) -> list[str]:
        """Expensive: recompute every balance from the stored postings."""
        st = world.state
        sums: dict[str, int] = {}
        for _, e in st.tables["acct_txns"].items():
            sums[e["acct"]] = sums.get(e["acct"], 0) + e["amount"]
        return [f"{a}: balance {r['balance']} != postings {sums.get(a, 0)}"
                for a, r in st.tables["accounts"].items() if r["balance"] != sums.get(a, 0)]

    def money_supply(self, world) -> dict:
        st = world.state
        sys_bal = {SYSTEM_ACCOUNTS[a]: st.tables["accounts"][a]["balance"] for a in SYSTEM_ACCOUNTS}
        private = sum(a["balance"] for a in st.tables["accounts"].values() if a["kind"] != "system")
        return {"system": sys_bal, "private": private}

    def observe(self, world, actor):
        st = world.state
        out = {}
        for owner in sorted(world.principals_of(actor)):
            for aid in st.find_ids("accounts", "owner", owner):
                out[aid] = st.tables["accounts"][aid]["balance"]
        return {"balances": out} if out else None
