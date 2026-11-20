"""Operator tools (host `admin`). Only the simulator operator (system principal) can call these; the kernel
rejects them for every in-world actor. They exist so that world setup, actor admission and experimental
interventions are recorded commands (replayable, regenerable), not out-of-band state edits."""
from __future__ import annotations

from ..core import schema as S
from ..core.errors import InvalidAction
from ..core.kernel import Module, endpoint, register_module
from .identity import HANDLE, ident


@register_module
class Admin(Module):
    name = "admin"
    depends = ("identity", "bank")

    @endpoint("join", S.obj({"handle": S.s_str(pattern=HANDLE, maxLength=31), "name": S.s_str(maxLength=120),
                             "kind": S.s_enum(["agent", "npc", "human"]), "endowment": S.s_int(minimum=0),
                             "password": S.s_str(maxLength=128), "controller": S.s_str(maxLength=120),
                             "info": S.s_any_obj()}, ["handle"]))
    def ep_join(self, ctx, handle, name=None, kind="agent", endowment=0, password=None, controller="", info=None):
        """Admit an actor: identity, a checking account, an optional endowment from the genesis account."""
        pid = ident(ctx.world).create_person(ctx, handle, name or handle, kind,
                                             dict(info or {}, controller=controller), password)
        bank = ctx.world.modules["bank"]
        aid = bank.open_account(ctx, pid, "checking")
        if endowment:
            bank.move(ctx, "acct_genesis", aid, endowment, "endowment", f"endowment for {pid}")
        hook = ctx.world.hooks.get("on_join")
        if hook:
            hook(ctx, pid, kind)
        return {"principal": pid, "account": aid}

    @endpoint("endow", S.obj({"principal": S.s_str(maxLength=60), "amount": S.s_int(minimum=1)},
                             ["principal", "amount"]))
    def ep_endow(self, ctx, principal, amount):
        bank = ctx.world.modules["bank"]
        aid = bank.resolve_account(ctx.state, principal)
        return {"txn": bank.move(ctx, "acct_genesis", aid, amount, "endowment", "operator endowment")}

    @endpoint("set_status", S.obj({"principal": S.s_str(maxLength=60), "status": S.s_enum(["active", "suspended"])},
                                  ["principal", "status"]))
    def ep_set_status(self, ctx, principal, status):
        if not ctx.state.has("principals", principal):
            raise InvalidAction("unknown principal")
        ctx.emit("principal.update", {"id": principal, "changes": {"status": status}}, vis=(principal,))
        return {"ok": True}

    @endpoint("broadcast", S.obj({"text": S.s_str(maxLength=2000), "kind": S.s_str(maxLength=40)}, ["text"]))
    def ep_broadcast(self, ctx, text, kind="news"):
        """Public announcement (e.g. an exogenous shock being announced). Shocks themselves are mechanisms."""
        ctx.emit("note", {"what": "broadcast", "kind": kind, "text": text}, vis="public")
        return {"ok": True}
