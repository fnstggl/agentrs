"""Participant seats (module `seat`): the principals a learner (or an evaluator-side reference) occupies in a Tier 2
domain world, created at genesis so that the world can be handed to any participant without operator calls.

Config: {"seats": [{"handle": "researcher", "name": "...", "endowment": cents, "kind": "agent"}]}. Each seat is an
ordinary person with a checking account funded from the genesis account (ledger kind `endowment`). The module has no
tools: a seat acts through the world's ordinary tools like everybody else.
"""
from __future__ import annotations

from ..core.kernel import Module, register_module


@register_module
class Seats(Module):
    name = "seat"
    depends = ("identity", "bank")

    def genesis(self, ctx):
        idm, bank = ctx.world.modules["identity"], ctx.world.modules["bank"]
        for s in self.config.get("seats") or []:
            pid = idm.create_person(ctx, s["handle"], s.get("name") or s["handle"], s.get("kind", "agent"),
                                    {"role": "participant", "controller": s.get("controller", "")})
            acct = bank.open_account(ctx, pid, "checking")
            if int(s.get("endowment", 0)):
                bank.move(ctx, "acct_genesis", acct, int(s["endowment"]), "endowment", f"endowment for {pid}")


def seat_principal(world, index: int = 0) -> str:
    """The principal of the world's `index`-th seat (evaluator and runner side)."""
    seats = world.modules["seat"].config.get("seats") or []
    return f"p:{seats[index]['handle']}"
