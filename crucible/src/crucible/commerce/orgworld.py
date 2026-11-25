"""World construction helpers for institution and labour worlds (used by the F2 tests and gate, and available to the
organisational-ecosystem and labour-marketplace domains).

Every helper acts through recorded commands (`system_call` for operator setup, `execute` for participants), so the
worlds they build replay and regenerate exactly.
"""
from __future__ import annotations

from ..core.kernel import World, WorldSpec

BASE_MODULES = ("identity", "bank", "admin", "contracts", "business", "gov", "deal", "inst", "commit")


def spec(seed: int, *, workers: int = 0, labor: dict | None = None, extra: tuple = (), params: dict | None = None,
         family: str = "f2_institutions") -> WorldSpec:
    """Identity, bank, operator tools, 4.0 contracts / companies / governance / negotiation, plus the 5.0
    institutions, commitments and labour market (with `workers` simulated workers)."""
    mods = [{"name": n, "config": {}} for n in BASE_MODULES]
    mods += [{"name": n, "config": {}} for n in extra]
    mods.append({"name": "labor", "config": dict(labor or {}, workers=int(workers))})
    return WorldSpec(family, int(seed), mods, dict(params or {}))


def create(seed: int, **kw) -> World:
    strict = kw.pop("strict", True)
    return World.create(spec(seed, **kw), strict=strict)


def act(w: World, actor: str, tool: str, /, **args):
    return w.execute(actor, {"tool": tool, "args": args})


def join(w: World, handle: str, cents: int = 0) -> tuple[str, str]:
    r = w.system_call("admin.join", {"handle": handle, "endowment": int(cents)})
    return r["principal"], r["account"]


def found_company(w: World, owner: str, handle: str, cash: int = 0, name: str | None = None,
                  kind: str = "company") -> tuple[str, str]:
    """`owner` founds an organisation, registers it as a company (paying the fee) and moves `cash` into its new
    account. Returns (org, account). Raises if any step is refused."""
    bank = w.modules["bank"]
    mine = bank.primary_account(w.state, owner)
    for tool, args in (("id.create_org", {"handle": handle, "name": name or handle, "kind": kind}),
                       ("biz.register", {"org": f"o:{handle}", "account": mine})):
        r = act(w, owner, tool, **args)
        if not r.ok:
            raise RuntimeError(f"{tool}: {r.error}")
    r = act(w, owner, "bank.open_account", owner=f"o:{handle}")
    if not r.ok:
        raise RuntimeError(f"bank.open_account: {r.error}")
    acct = r.result["account"]
    if cash:
        r = act(w, owner, "bank.transfer", from_account=mine, to=acct, amount=int(cash))
        if not r.ok:
            raise RuntimeError(f"bank.transfer: {r.error}")
    return f"o:{handle}", acct


def add_member(w: World, org: str, by: str, person: str, role: str = "member") -> None:
    """`by` invites `person`, who accepts (membership is accepted, never assigned)."""
    r = act(w, by, "id.invite", org=org, person=person, role=role)
    if not r.ok:
        raise RuntimeError(f"id.invite: {r.error}")
    r = act(w, person, "id.respond_invite", invite=r.result["invite"], accept=True)
    if not r.ok:
        raise RuntimeError(f"id.respond_invite: {r.error}")


def balance(w: World, principal: str) -> int:
    return sum(a["balance"] for a in w.state.find("accounts", "owner", principal))


def verify_reproducible(w: World) -> dict:
    """Replay (appliers, hash chain, state digest) and regeneration (handlers re-run from the recorded commands)
    of a world built in memory from genesis."""
    evs = [e.to_dict() for e in w.log]
    rep = World.replay(w.spec, evs)
    gen = World.regenerate(w.spec, evs)
    return {"replay_digest_equal": rep.digest() == w.digest(), "regeneration_hash_equal": gen.last_hash == w.last_hash,
            "events": len(evs)}
