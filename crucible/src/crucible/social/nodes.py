"""Social node ids: one compact integer per principal for every social structure (kernel module `social_ids`).

The social graph, community memberships, conversations and the content catalogue store principals as int64 node
ids so that population-scale structures are dense arrays:

* a population individual `p<i>` (crucible.substrate.ids) is node `i` - no record, no event;
* any other principal (a 4.0 participant `p:<handle>`, an organisation `o:<handle>`, a service `svc:<host>`) gets
  node `OFFSET + k` the first time a social module needs it (`ensure(ctx, principal)` emits `social.node`).

`OFFSET` = 2**40 leaves room for any population size. `node_of(world, principal)` is the read-only lookup (None when
the principal has no node yet), `principal_of(world, node)` the inverse, `principals_of_nodes` the vectorised one.
"""
from __future__ import annotations

import numpy as np

from ..core import schema as S
from ..core.errors import InvalidAction, NotFound
from ..core.kernel import SYSTEM, Ctx, Module, applier, register_module
from ..substrate import bus, ids

OFFSET = 1 << 40


def is_pop(node: int) -> bool:
    return 0 <= int(node) < OFFSET


def node_of(world, principal: str) -> int | None:
    i = ids.index(principal)
    if i is not None:
        return i
    rec = world.state.tables.get("social_nodes", {}).get(principal)
    return None if rec is None else int(rec["node"])


def ensure(ctx: Ctx, principal: str) -> int:
    """The node of `principal`, creating it (one `social.node` event) the first time."""
    i = ids.index(principal)
    if i is not None:
        return i
    if not isinstance(principal, str) or principal == SYSTEM or not ctx.state.has("principals", principal):
        raise NotFound(f"no principal {principal}")
    rec = ctx.state.get("social_nodes", principal)
    if rec is not None:
        return int(rec["node"])
    k = ctx.state.next_counter("social_node") - 1
    node = OFFSET + k
    ctx.emit("social.node", {"principal": principal, "node": node}, vis="internal")
    return node


def principal_of(world, node: int) -> str | None:
    node = int(node)
    if is_pop(node):
        return ids.pid(node)
    rec = world.state.tables.get("social_node_ids", {}).get(str(node))
    return None if rec is None else rec["principal"]


def principals_of_nodes(world, nodes) -> list[str | None]:
    return [principal_of(world, int(n)) for n in np.asarray(nodes).reshape(-1)]


def require(world, principal: str) -> int:
    n = node_of(world, principal)
    if n is None:
        raise NotFound(f"{principal} has no social presence yet")
    return n


bus.event_type("social.node", S.obj({"principal": {"type": "string", "minLength": 1, "maxLength": 120},
                                     "node": {"type": "integer", "minimum": OFFSET}}, ["principal", "node"]),
               owner="social_ids", doc="a non-population principal received its social node id")


@register_module
class SocialIds(Module):
    """Owner of the principal <-> node mapping for principals that are not population rows. No tools."""
    name = "social_ids"
    host = "social_ids"
    version = "1"

    def install(self, world):
        world.state.declare_table("social_nodes")          # principal -> {node}
        world.state.declare_table("social_node_ids")       # str(node) -> {principal}
        bus.enforce(world, owner="social_ids")

    @applier("social.node")
    def ap_node(self, ev):
        p = ev.payload
        st = self.world.state
        if st.has("social_nodes", p["principal"]):
            raise InvalidAction(f"{p['principal']} already has a node")
        st.insert("social_nodes", p["principal"], {"node": p["node"], "t": ev.t})
        st.insert("social_node_ids", str(p["node"]), {"principal": p["principal"]})

    def check_invariants(self, world):
        errs = []
        st = world.state
        for pr, rec in st.tables["social_nodes"].items():
            back = st.tables["social_node_ids"].get(str(rec["node"]))
            if back is None or back["principal"] != pr:
                errs.append(f"node mapping of {pr} is not reversible")
            if rec["node"] < OFFSET:
                errs.append(f"{pr} has a population node id")
        if len(st.tables["social_nodes"]) != len(st.tables["social_node_ids"]):
            errs.append("node mapping tables differ in size")
        return errs
