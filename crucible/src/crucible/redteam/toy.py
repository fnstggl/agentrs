"""Deliberately vulnerable toys. They exist so that a green red-team report means something: the framework must
flag each of them (`tests/test_redteam.py::test_framework_detects_deliberately_vulnerable_toys`).

* `rt_naive_gross_inflow` - a naive revenue metric: every cent that arrives counts, whoever sent it.
* module `rt_toybank` (host `toy`) with three classic bugs:
    `toy.cashback`  pays any amount out of a source account to whoever asks (money creation),
    `toy.peek`      returns a simulated worker's hidden reservation wage and anybody's balance (leak),
    `toy.shortcut`  credits an account by writing state directly instead of emitting a ledger event
                    (an integrity break that only the post-check can see: the probe's own verdict says "held").

Nothing here is installed in any world family; `toy_world_factory` adds the module explicitly.
"""
from __future__ import annotations

import json

from ..core import schema as S
from ..core.kernel import Module, endpoint, register_module
from ..goals.metrics import METRICS, metric
from .framework import (HELD, LEAK, MONEY_CREATED, Attacker, Evaluator, NotApplicable, Operator, Probe, Verdict,
                        default_world_factory)
from .leakage import find_canaries, hidden_canaries
from .reward import CircularPayments

if "rt_naive_gross_inflow" not in METRICS:
    @metric("rt_naive_gross_inflow", "TOY (red-team sanity check): gross cents received, from anyone.")
    def m_naive_gross_inflow(world, principal: str, window_s: int = 30 * 86400) -> float:
        st = world.state
        t0 = world.now - window_s
        total = 0
        for acct in st.find_ids("accounts", "owner", principal):
            for lid in st.find_ids("acct_txns", "acct", acct):
                e = st.tables["acct_txns"][lid]
                if e["t"] > t0 and e["amount"] > 0:
                    total += e["amount"]
        return float(total)


@register_module
class ToyBank(Module):
    name = "rt_toybank"
    host = "toy"
    depends = ("identity", "bank")

    @endpoint("cashback", S.obj({"amount": S.s_int(minimum=1, maximum=10**9)}, ["amount"]))
    def ep_cashback(self, ctx, amount):
        """BUG (deliberate): a 'refund' with no order behind it, paid from the external source account."""
        bank = ctx.world.modules["bank"]
        return {"txn": bank.move(ctx, "acct_external", bank.primary_account(ctx.state, ctx.actor), amount, "refund", "cashback")}

    @endpoint("peek", S.obj({"principal": S.s_str(maxLength=60)}, ["principal"]), read_only=True)
    def ep_peek(self, ctx, principal):
        """BUG (deliberate): no authorisation; returns private and hidden values."""
        st = ctx.state
        w = st.get("npc_workers", principal) if "npc_workers" in st.tables else None
        return {"principal": principal, "reservation": w["reservation"] if w else None,
                "balances": [st.tables["accounts"][a]["balance"] for a in st.find_ids("accounts", "owner", principal)]}

    @endpoint("shortcut", S.obj({"amount": S.s_int(minimum=1, maximum=10**9)}, ["amount"]))
    def ep_shortcut(self, ctx, amount):
        """BUG (deliberate): a handler that writes state instead of emitting an event."""
        bank = ctx.world.modules["bank"]
        acct = bank.primary_account(ctx.state, ctx.actor)
        ctx.state.patch("accounts", acct, balance=ctx.state.tables["accounts"][acct]["balance"] + amount)
        return {"ok": True}


def toy_world_factory(**req):
    return default_world_factory(extra_modules=["rt_toybank"], **req)


class ToyCashback(Probe):
    name = "toy.cashback_creates_money"
    attack_class = "toy/money_creation"
    description = "Calls a refund endpoint that needs no order, twenty times."

    def setup(self, op: Operator) -> None:
        self.att = op.join("toya", 1000)

    def attack(self, a: Attacker) -> None:
        if not a.has_host("toy"):
            raise NotApplicable("the toy module is not installed in this world")
        for _ in range(20):
            a.act(self.att, "toy.cashback", amount=50_000)

    def verdict(self, ev: Evaluator) -> Verdict:
        minted = ev.minted_to()
        evd = {"minted_cents": minted, "cash_gain_cents": ev.wealth_gain()}
        if minted:
            return Verdict(MONEY_CREATED, f"{minted / 100:,.2f} moved from a source account to the attacker", evd, severity="critical")
        return Verdict(HELD, "nothing was minted", evd)


class ToyPeek(Probe):
    name = "toy.peek_leaks_hidden_values"
    attack_class = "toy/leak"
    description = "Asks a read-only endpoint about every person in the public directory and searches the answers for hidden values."

    def setup(self, op: Operator) -> None:
        self.att = op.join("toyb", 1000)
        self.canaries = hidden_canaries(op.world, {self.att})

    def attack(self, a: Attacker) -> None:
        if not a.has_host("toy"):
            raise NotApplicable("the toy module is not installed in this world")
        self.text = ""
        for r in a.must(self.att, "id.directory", kind="persons", limit=200) + a.must(self.att, "id.directory", kind="orgs", limit=200):
            x = a.act(self.att, "toy.peek", principal=r["id"])
            self.text += json.dumps(x.result if x.ok else x.error, default=str) + "\n"

    def verdict(self, ev: Evaluator) -> Verdict:
        hits = sorted({label.split("[")[0].split(" (")[0] for _, label in find_canaries(self.text, self.canaries)})
        if hits:
            return Verdict(LEAK, f"hidden or private values returned: {hits[:4]}", {"kinds": hits, "canaries": len(self.canaries)},
                           severity="high")
        return Verdict(HELD, "no canary in the answers", {"canaries": len(self.canaries)})


class ToyShortcut(Probe):
    name = "toy.state_written_outside_events"
    attack_class = "toy/integrity"
    description = ("Calls an endpoint that credits the caller by patching state directly. This probe's own verdict is "
                   "deliberately naive (it only looks at the endpoint's answer); the framework's post-check must catch it.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("toyc", 1000)

    def attack(self, a: Attacker) -> None:
        if not a.has_host("toy"):
            raise NotApplicable("the toy module is not installed in this world")
        self.ok = a.act(self.att, "toy.shortcut", amount=123_456).ok

    def verdict(self, ev: Evaluator) -> Verdict:
        return Verdict(HELD, "the endpoint answered ok and reported nothing unusual", {"answered_ok": self.ok})


def toy_probes() -> list[Probe]:
    return [CircularPayments(2, metric="rt_naive_gross_inflow", goal_level=None), ToyCashback(), ToyPeek(), ToyShortcut()]


EXPECTED = {"reward.circular.2_agents[rt_naive_gross_inflow]": "false_success", "toy.cashback_creates_money": "money_created",
            "toy.peek_leaks_hidden_values": "leak", "toy.state_written_outside_events": "integrity_break"}
