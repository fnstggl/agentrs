"""Shared world primitives: one owner per shared fact (Crucible 5.0, contract 1.5; MASTER_SPEC Part 2.3).

`OWNERS` maps each shared kind of fact to the ONE kernel module and state table that is its source of truth, and to
the API through which other modules read and change it. Other modules reference these facts by id (a principal, an
account, a REF, a repository id...) and never keep their own copy. The registry is maintained by workstream A only:
new kinds and changes of ownership are added here.

* `owner(kind)` -> {"module", "table", "doc", "api", "ids", "since"} ("@..." tables are kernel structures that are
  not state tables: the hash-chained event log and per-event visibility).
* `kinds()` -> every kind; `implemented()` -> kinds whose owner module is a registered kernel module (or the kernel).
* A module that holds a shared fact declares it with a class attribute `shared_facts = {kind: table}`;
  `check_no_duplicates(world)` raises InvariantViolation when
    1. a module declares a kind it does not own (two modules for one kind),
    2. a module declares a kind that does not exist (only workstream A adds kinds),
    3. two modules declare the same state table (both would write the same facts),
    4. the owner module of a kind is in the world but its table is missing.
  It returns a report (kinds present, owners, and the recognised 4.0 overlaps as warnings) when there is none.
"""
from __future__ import annotations

from ..core.errors import InvariantViolation, ProtocolError

KERNEL = "kernel"

#: kind -> owner. `since`: the Crucible version that introduced the owner table.
OWNERS: dict[str, dict] = {
    "identity": {"module": "identity", "table": "persons", "since": "1",
                 "ids": "p:<handle> (participants); p<i> (population individuals, crucible.substrate.ids)",
                 "api": "identity.create_person / substrate.ids.ensure_principal; kernel table `principals` holds the "
                        "authentication record of every principal (kind, status) under the same id",
                 "doc": "participant identities"},
    "organization": {"module": "identity", "table": "orgs", "since": "1", "ids": "o:<handle>",
                     "api": "identity.create_org, id.invite/respond_invite/set_role (members table)",
                     "doc": "organisations and their members and roles"},
    "account": {"module": "bank", "table": "accounts", "since": "1", "ids": "acct_<n>, acct_<system role>",
                "api": "bank.open_account, bank.primary_account", "doc": "money accounts (owner, kind, balance, floor)"},
    "money": {"module": "bank", "table": "txns", "since": "1", "ids": "tx_<n>",
              "api": "bank.move / bank.post (double-entry, every movement one bank.txn event)",
              "doc": "money and its movements (integer cents, conserved)"},
    "asset": {"module": "assets", "table": "assets", "since": "5",
              "ids": "ast_<n>", "api": "crucible.substrate.assets (asset.register/transfer/offer/accept)",
              "doc": "ownable, transferable non-money assets (licences, rights, digital goods, titles); specific asset "
                     "classes keep their 4.0 owners: equity business.shares, physical capital industry.machines, "
                     "design rights industry.design_rights, domains net.domains"},
    "resource_inventory": {"module": "industry", "table": "stock", "since": "4", "ids": "<facility>|<sku>",
                           "api": "industry stock events (make.stock_add / make.stock_take)",
                           "doc": "physical resources and goods held at facilities (conserved)"},
    "content_artifact": {"module": "artifacts", "table": "artifacts", "since": "5", "ids": "sha256:<64 hex>",
                         "api": "crucible.substrate.artifacts.put/get/info/publish; tools artifact.*",
                         "doc": "immutable, content-addressed content (text, images, audio, video, code, data)"},
    "product": {"module": "market", "table": "listings", "since": "2", "ids": "lst_<n>",
                "api": "market.list/buy/deliver (goods, services, API plans offered for sale)",
                "doc": "products and offers sold to customers"},
    "service": {"module": "cloud", "table": "hosts", "since": "2", "ids": "<host>",
                "api": "cloud.rent/deploy (deployments, instances); service principal svc:<host>",
                "doc": "running services reachable as hosts"},
    "software_repository": {"module": "code", "table": "repos", "since": "2", "ids": "<owner handle>/<name>",
                            "api": "code.create_repo/commit/fork (commits, blobs)", "doc": "source repositories"},
    "relationship": {"module": "forum", "table": "forum_follows", "since": "2", "ids": "<follower>|<target>",
                     "api": "forum.follow/unfollow",
                     "doc": "directed social ties between principals (follows); organisation membership is part of "
                            "`organization`, contract parties part of `contract`"},
    "community": {"module": "forum", "table": "forum_boards", "since": "2", "ids": "<board name>",
                  "api": "Forum.create_board (genesis/config), forum.post/reply/feed", "doc": "communities (discussion boards)"},
    "institution": {"module": "gov", "table": "constitutions", "since": "3", "ids": "<org>",
                    "api": "gov.adopt/amend/propose/vote (proposals, votes)",
                    "doc": "institutional rules of organisations (constitutions, decisions)"},
    "contract": {"module": "contracts", "table": "contracts", "since": "1", "ids": "ctr_<n>",
                 "api": "contracts.propose/sign/terminate (parties, clauses, claims)", "doc": "binding agreements"},
    "knowledge_artifact": {"module": "docs", "table": "documents", "since": "2", "ids": "doc_<n>",
                           "api": "docs.create/edit/share (versions)",
                           "doc": "editable knowledge documents (versioned); immutable published bytes are "
                                  "content_artifact"},
    "tool_definition": {"module": "caps", "table": "capabilities", "since": "3", "ids": "cap_<n>",
                        "api": "caps.publish/search/install; built-in tools are kernel endpoints (world.find_tools)",
                        "doc": "declared capabilities of agent-created services (descriptors, schemas)"},
    "time": {"module": KERNEL, "table": "sched", "since": "1", "ids": "job_<n>",
             "api": "World.advance_time, ctx.schedule; the clock is state.meta['clock']",
             "doc": "the simulated clock and scheduled processes"},
    "provenance": {"module": KERNEL, "table": "@log", "since": "1", "ids": "event seq",
                   "api": "hash-chained event log (World.events_range, analytics.world_events); artifact lineage "
                          "in artifacts.parents; recorded inference in crucible.substrate.inference",
                   "doc": "who did what, when, and what it derived from"},
    "permission": {"module": "identity", "table": "grants", "since": "1", "ids": "grant_<n>",
                   "api": "identity.authorize / id.grant / id.revoke; artifact grants in artifact_grants",
                   "doc": "delegated capabilities with scopes, limits and expiry"},
    "visibility": {"module": KERNEL, "table": "@vis", "since": "1", "ids": "event vis",
                   "api": "Event.vis (principals | 'public' | 'internal'), World.visible, hooks['principals_of']",
                   "doc": "who can observe each recorded fact"},
}

#: 4.0 tables that hold facts of an owned kind in a separate namespace. Reported as warnings by
#: check_no_duplicates when both modules are in a world; no new module may add another.
OVERLAPS: dict[str, list[dict]] = {
    "resource_inventory": [{"module": "market", "table": "inventory",
                            "why": "4.0 marketplace goods ledger (owner|sku), minted and conserved inside the market; "
                                   "industrial stock (facility|sku) is a separate namespace, never synchronised"}],
}

#: tables kernel structures own; modules never declare them
_KERNEL_TABLES = ("principals", "sched", "idem", "knowledge")

# ---------------------------------------------------------------------------------------------------- typed events
# The events that create or move the shared facts of the 4.0 owners, registered with their payload schemas (the
# shapes the 4.0 modules emit; extra keys are tolerated so that 4.0 logs keep validating). The 5.0 owners register
# their own (artifact.*, asset.*, cols.*).
_S, _I, _O = {"type": "string"}, {"type": "integer"}, {"type": "object", "additionalProperties": True}
_SN, _IN, _ON = {"type": ["string", "null"]}, {"type": ["integer", "null"]}, \
    {"type": ["object", "null"], "additionalProperties": True}


def _ev(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": True}


PRIMITIVE_EVENTS: dict[str, tuple[str, str, dict]] = {
    # type: (kind, owner module, schema)
    "principal.add": ("identity", KERNEL, _ev({"id": _S, "kind": {"type": "string",
                                                                   "enum": ["person", "org", "service", "system"]},
                                                "info": _O, "status": _S}, ["id", "kind", "info"])),
    "id.person": ("identity", "identity", _ev({"id": _S, "handle": _S, "name": _S, "kind": _S, "info": _O},
                                               ["id", "handle", "name", "kind", "info"])),
    "id.org": ("organization", "identity", _ev({"id": _S, "handle": _S, "name": _S, "kind": _S, "founder": _SN,
                                                "info": _O}, ["id", "handle", "name", "kind", "founder", "info"])),
    "id.member": ("organization", "identity", _ev({"org": _S, "person": _S, "role": {"type": "string", "enum": [
        "owner", "admin", "member"]}, "title": _S}, ["org", "person", "role"])),
    "id.grant": ("permission", "identity", _ev({"id": _S, "grantee": _S, "owner": _S, "cap": _S, "scope": _SN,
                                                "limit": _IN, "expires": _IN, "by": _S},
                                               ["id", "grantee", "owner", "cap", "scope", "limit", "expires", "by"])),
    "bank.account": ("account", "bank", _ev({"id": _S, "owner": _S, "kind": _S, "name": _S, "overdraft": _I},
                                            ["id", "owner", "kind", "name", "overdraft"])),
    "bank.txn": ("money", "bank", _ev({"id": _S, "postings": {"type": "array", "minItems": 1, "items": {
        "type": "array", "minItems": 2, "maxItems": 2}}, "kind": _S, "memo": _S, "ref": _SN, "by": _S},
        ["id", "postings", "kind", "memo", "ref", "by"])),
    "code.repo": ("software_repository", "code", _ev({"id": _S, "owner": _S, "visibility": _S, "description": _S,
                                                      "license": _S}, ["id", "owner", "visibility"])),
    "cloud.deployment": ("service", "cloud", _ev({"id": _S, "host": _S, "owner": _S, "repo": _S, "commit": _S,
                                                  "instance": _S, "manifest": _O, "svc": _S},
                                                 ["id", "host", "owner", "repo", "commit"])),
    "forum.board": ("community", "forum", _ev({"id": _S, "title": _S, "description": _S}, ["id", "title"])),
    "forum.follow": ("relationship", "forum", _ev({"id": _S, "follower": _S, "target": _S},
                                                  ["id", "follower", "target"])),
    "docs.create": ("knowledge_artifact", "docs", _ev({"id": _S, "owner": _S, "title": _S, "body": _S,
                                                       "visibility": _S, "by": _S}, ["id", "owner", "title"])),
    "gov.constitution": ("institution", "gov", _ev({"org": _S, "rules": {"type": "array"}, "distribution": _ON},
                                                   ["org", "rules"])),
    "market.listing": ("product", "market", _ev({"id": _S, "seller": _S, "kind": _S, "title": _S, "description": _S,
                                                 "price": {"type": "integer", "minimum": 0}, "host": _SN,
                                                 "endpoint": _SN, "need": _SN, "repo": _SN, "sku": _SN},
                                                ["id", "seller", "kind", "title", "price"])),
    "contracts.proposed": ("contract", "contracts", _ev({"id": _S, "proposer": _S, "parties": {
        "type": "array", "items": _S, "minItems": 1}, "clauses": {"type": "array"}, "title": _S, "text": _S,
        "expires": _I, "employee": _SN, "employer": _SN}, ["id", "proposer", "parties", "clauses"])),
    "contracts.signed": ("contract", "contracts", _ev({"id": _S, "party": _S, "accounts": _O}, ["id", "party"])),
    "contracts.status": ("contract", "contracts", _ev({"id": _S, "status": _S, "reason": _SN}, ["id", "status"])),
    "caps.published": ("tool_definition", "caps", _ev({"id": _S, "provider": _S, "host": _S, "title": _S,
                                                       "params": _O, "requires": {"type": "array"},
                                                       "endpoints": _O, "commit": _S},
                                                      ["id", "provider", "host", "endpoints"])),
    "sched.add": ("time", KERNEL, _ev({"id": _S, "at": {"type": "integer", "minimum": 0}, "prio": _I, "n": _I,
                                       "proc": _S, "args": _O, "period": _IN},
                                      ["id", "at", "prio", "n", "proc", "args", "period"])),
    "make.stock_add": ("resource_inventory", "industry", _ev({"fac": _S, "sku": _S, "qty": {"type": "integer",
                                                                                            "minimum": 0}},
                                                             ["fac", "sku", "qty"])),
    "make.stock_take": ("resource_inventory", "industry", _ev({"fac": _S, "sku": _S, "qty": {"type": "integer",
                                                                                             "minimum": 0}},
                                                              ["fac", "sku", "qty"])),
}


def register_owner_events() -> int:
    """Register the payload schemas of the events that create or move the 4.0 owners' shared facts (bank.txn,
    id.person, ...). Called explicitly when the kernel modules are loaded (`crucible.load_all_modules` imports
    `crucible.substrate.modules`, which calls it) and by `bus.enforce_world`; idempotent. Importing this module
    registers nothing (5.0 audit A5: registration must not depend on who happened to import what)."""
    from . import bus
    for name, (kind, mod, schema) in PRIMITIVE_EVENTS.items():
        bus.event_type(name, schema, owner=mod, doc=f"4.0 {kind} fact ({OWNERS[kind]['module']}."
                                                     f"{OWNERS[kind]['table']})")
    return len(PRIMITIVE_EVENTS)


def kinds() -> list[str]:
    return list(OWNERS)


def owner(kind: str) -> dict:
    """{"module", "table", ...} of the one owner of `kind`."""
    o = OWNERS.get(kind)
    if o is None:
        raise ProtocolError(f"unknown shared kind {kind!r} (kinds: {', '.join(OWNERS)})")
    return dict(o)


def implemented() -> list[str]:
    """Kinds whose owner is the kernel or a registered kernel module (call crucible.load_all_modules() first)."""
    from ..core.kernel import MODULES
    return [k for k, o in OWNERS.items() if o["module"] == KERNEL or o["module"] in MODULES]


def owner_modules() -> list[str]:
    return sorted({o["module"] for o in OWNERS.values() if o["module"] != KERNEL})


def declared(world) -> dict[str, list[tuple[str, str]]]:
    """kind -> [(module, table)] declared in `world` (OWNERS for owner modules present, plus `shared_facts`)."""
    out: dict[str, list[tuple[str, str]]] = {}
    for kind, o in OWNERS.items():
        if o["module"] == KERNEL or o["module"] in world.modules:
            out.setdefault(kind, []).append((o["module"], o["table"]))
    for name, mod in world.modules.items():
        for kind, table in (getattr(type(mod), "shared_facts", None) or {}).items():
            pair = (name, table)
            if pair not in out.setdefault(kind, []):
                out[kind].append(pair)
    return out


def problems(world) -> list[str]:
    """Everything check_no_duplicates would fail on (empty = no duplicate shared facts)."""
    errs = []
    for name, mod in sorted(world.modules.items()):
        for kind, table in sorted((getattr(type(mod), "shared_facts", None) or {}).items()):
            o = OWNERS.get(kind)
            if o is None:
                errs.append(f"module {name} declares unknown shared kind {kind!r} (only workstream A adds kinds)")
            elif o["module"] != name or o["table"] != table:
                errs.append(f"duplicate {kind}: module {name} declares table {table!r}, but {kind} is owned by "
                            f"{o['module']}.{o['table']}")
    for kind, pairs in sorted(declared(world).items()):
        if len(pairs) > 1:
            errs.append(f"duplicate {kind}: declared by {', '.join(f'{m}.{t}' for m, t in pairs)}")
    for table, mods in sorted(world.state.declared_by.items()):
        if len(mods) > 1:
            errs.append(f"table {table!r} is declared by {len(mods)} modules ({', '.join(mods)})")
        elif table in _KERNEL_TABLES and mods != ["kernel"]:
            errs.append(f"kernel table {table!r} declared by module {mods[0]}")
    for kind, o in OWNERS.items():
        if o["module"] in world.modules and not o["table"].startswith("@") and o["table"] not in world.state.tables:
            errs.append(f"owner of {kind} ({o['module']}) is in the world but table {o['table']!r} is missing")
        if o["module"] == KERNEL and not o["table"].startswith("@") and o["table"] not in world.state.tables:
            errs.append(f"kernel table {o['table']!r} of {kind} is missing")
    return errs


def check_no_duplicates(world) -> dict:
    """Raise InvariantViolation if the world keeps a shared fact in more than one place; else return the report
    {kinds: {kind: {module, table, present}}, present: n, warnings: [...]}."""
    errs = problems(world)
    if errs:
        raise InvariantViolation("duplicate shared facts: " + "; ".join(errs))
    rep = {}
    for kind, o in OWNERS.items():
        present = o["module"] == KERNEL or o["module"] in world.modules
        rep[kind] = {"module": o["module"], "table": o["table"], "present": present}
    warnings = []
    for kind, lst in OVERLAPS.items():
        o = OWNERS[kind]
        for ov in lst:
            if o["module"] in world.modules and ov["module"] in world.modules:
                warnings.append(f"{kind}: {ov['module']}.{ov['table']} ({ov['why']})")
    return {"kinds": rep, "present": sum(1 for r in rep.values() if r["present"]), "warnings": warnings}
