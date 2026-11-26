"""Identity indices (Crucible 5.0, contract 1.3).

Population individuals are integer rows `0..N-1` of the column tables. An individual becomes a kernel principal only
when it needs one (a bank account, a membership, a message): `ensure_principal(ctx, i)` creates principal `p<i>`
(kernel kind `person`) and its identity person record on first use, and is idempotent. The principal does not copy
any column state: its person record holds only the row index (`info.pop_index`), so the individual's state has one
source of truth, the columns.

4.0 participants keep their `p:<handle>` ids; `p<i>` (no colon) cannot collide with them, and bare handles never
contain a colon.
"""
from __future__ import annotations

import re

from ..core.errors import Conflict, InvalidAction
from ..core.kernel import Ctx

_PID = re.compile(r"p(0|[1-9][0-9]{0,11})")
PERSON_KIND = "population"


def pid(i: int) -> str:
    """The principal id of individual `i`: "p<i>"."""
    i = int(i)
    if i < 0:
        raise InvalidAction("individual index must be >= 0")
    return f"p{i}"


def index(principal: str) -> int | None:
    """The individual index of a principal "p<i>", else None (e.g. "p:alice", "o:acme", "p01")."""
    if not isinstance(principal, str):
        return None
    m = _PID.fullmatch(principal)
    return int(m.group(1)) if m else None


def ensure_principal(ctx: Ctx, i: int, kind: str = "person", *, account: bool = False, name: str | None = None) -> str:
    """Create (once) the kernel principal and identity person record of individual `i`; returns "p<i>".

    `kind` is the kernel principal kind ("person": may act; "service" is refused - individuals are persons).
    With `account=True` the individual also gets a checking account in `bank` (once), so its payments are real
    ledger moves."""
    if kind != "person":
        raise InvalidAction("population individuals are principals of kind 'person'")
    p = pid(i)
    st = ctx.state
    if not st.has("principals", p):
        ctx.emit("principal.add", {"id": p, "kind": "person", "info": {"name": name or f"individual {int(i)}",
                                                                       "pop_index": int(i)}}, vis="internal")
    elif st.tables["principals"][p]["kind"] != "person":
        raise Conflict(f"{p} exists and is not a person")
    if "persons" in st.tables and not st.has("persons", p):
        if "identity" not in ctx.world.modules:
            raise InvalidAction("persons table without the identity module")
        ctx.emit("id.person", {"id": p, "handle": p, "name": name or f"individual {int(i)}", "kind": PERSON_KIND,
                               "info": {"pop_index": int(i)}}, vis="internal")
    if account:
        bank = ctx.world.modules.get("bank")
        if bank is None:
            raise InvalidAction("account=True needs the bank module")
        if bank.primary_account(st, p) is None:
            bank.open_account(ctx, p, "checking")
    return p


def principal_of(world, i: int) -> str | None:
    """"p<i>" if individual `i` has been materialised as a principal, else None."""
    p = pid(i)
    return p if world.state.has("principals", p) else None


def materialised(world) -> list[int]:
    """Indices of individuals that are kernel principals (sorted)."""
    out = []
    for p in world.state.find_ids("principals", "kind", "person"):
        i = index(p)
        if i is not None:
            out.append(i)
    return sorted(out)
