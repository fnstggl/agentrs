"""Typed cross-module events (Crucible 5.0, MASTER_SPEC Part 2.3: "systems communicate through typed events").

* `event_type(name, schema, owner, doc)` registers an event type `<host>.<what>` with a payload schema (the kernel's
  JSON-Schema subset, `crucible.core.schema`) and the module that owns it. Registration is global and idempotent;
  a conflicting second definition is a ProtocolError.
* `emit(ctx, name, payload, vis)` emits a registered type; in strict worlds (`World(strict=True)` or
  `params["typed_events"]`) the payload is validated first, and an invalid payload is a ProtocolError.
* `subscribe(world, name, fn, owner)` registers a same-transaction subscriber (in `Module.install`). The kernel calls
  `fn(ctx, event)` right after the event's applier, inside the emitting transaction, as the system principal: it may
  read state, emit events and call tools; if it raises, the whole command rolls back (unless the emitting handler
  contains the error itself). Replay never calls subscribers (their events are in the log); regeneration re-runs them,
  so they must be deterministic. Recursion depth is bounded by the kernel (8 nested calls/subscriptions).
* `enforce(world, owner=None)` makes a strict world validate *every* emitted event of the registered types (also
  those emitted with plain `ctx.emit`), not only those going through `bus.emit`.

With no subscriber registered the kernel behaves exactly as 4.0 (one empty-dict test per emit).
"""
from __future__ import annotations

import re
from typing import Callable

from ..core import schema as S
from ..core.errors import ProtocolError

NAME_RE = re.compile(r"[a-z][a-z0-9_]*(\.[a-z0-9_]+)+")
# kernel event types emitted at commit or in failure paths: a subscriber there could not roll anything back
FORBIDDEN = frozenset({"ctr", "ext", "idem", "cmd", "cmd.rejected", "proc", "proc.failed", "sched.add", "sched.del"})

_TYPES: dict[str, dict] = {}


def event_type(name: str, schema: dict, owner: str, doc: str = "") -> None:
    """Register event type `name` ("<host>.<what>") with its payload schema and owning module."""
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise ProtocolError(f"bad event type name {name!r} (expected '<host>.<what>')")
    errs = S.check_schema(schema)
    if errs:
        raise ProtocolError(f"event type {name}: bad schema: {errs[:3]}")
    if not owner or not isinstance(owner, str):
        raise ProtocolError(f"event type {name}: owner module required")
    rec = {"schema": schema, "owner": owner, "doc": doc}
    old = _TYPES.get(name)
    if old is not None:
        if old["schema"] != schema or old["owner"] != owner:
            raise ProtocolError(f"event type {name} already registered by {old['owner']} with another schema")
        if doc and not old["doc"]:
            old["doc"] = doc
        return
    _TYPES[name] = rec


def types() -> dict[str, dict]:
    """name -> {schema, owner, doc} of every registered event type."""
    return {k: dict(v) for k, v in sorted(_TYPES.items())}


def validate(name: str, payload: dict) -> list[str]:
    t = _TYPES.get(name)
    if t is None:
        return [f"unregistered event type {name}"]
    return S.validate(t["schema"], payload, path=name)


def _strict(world) -> bool:
    return bool(world.strict or world.spec.params.get("typed_events"))


def emit(ctx, name: str, payload: dict, vis=None):
    """Emit a registered event type through the kernel (validated in strict worlds). Returns the Event."""
    if name not in _TYPES:
        raise ProtocolError(f"unregistered event type {name} (register it with bus.event_type)")
    if _strict(ctx.world):
        errs = S.validate(_TYPES[name]["schema"], payload, path=name)
        if errs:
            raise ProtocolError(f"invalid {name} payload: {'; '.join(errs[:5])}")
    return ctx.emit(name, payload, vis)


def subscribe(world, name: str, fn: Callable, owner: str) -> None:
    """Call `fn(ctx, event)` in the emitting transaction right after every `name` event is applied."""
    if not isinstance(name, str) or not NAME_RE.fullmatch(name) or name in FORBIDDEN:
        raise ProtocolError(f"cannot subscribe to {name!r}")
    if not owner or owner not in world.modules and owner != "kernel":
        raise ProtocolError(f"subscriber owner {owner!r} is not a module of this world")
    subs = world.subscribers.setdefault(name, [])
    if any(o == owner and f == fn for o, f in subs):
        return
    subs.append((owner, fn))


def subscribers(world, name: str | None = None) -> dict[str, list[str]]:
    """Registered subscriber owners by event type (for inspection)."""
    return {k: [o for o, _ in v] for k, v in sorted(world.subscribers.items()) if name is None or k == name}


def enforce(world, owner: str | None = None) -> int:
    """In a strict world, validate every emitted event of the registered types (of `owner`, or all) after it is
    applied; an invalid one raises ProtocolError (the transaction rolls back and a strict world re-raises).
    Returns the number of types enforced (0 in a non-strict world)."""
    if not _strict(world):
        return 0
    n = 0
    for name, t in sorted(_TYPES.items()):
        if owner is not None and t["owner"] != owner:
            continue
        if name in FORBIDDEN:      # kernel commit/failure-path types cannot be subscribed to (validated by tests)
            continue
        subscribe(world, name, _validator(name), owner="kernel")
        n += 1
    return n


def enforce_world(world) -> int:
    """Typed worlds (`params["typed_events"]`): validate every event of every registered type owned by a module of
    this world (or the kernel), the 4.0 owners' shared-fact events included. Called by the kernel when such a world
    is constructed. Validators are passive same-transaction subscribers: they draw no randomness and emit nothing,
    so a typed world runs exactly the trajectory of the same world untyped (D-21) - an invalid payload is the only
    thing that changes the outcome (the command is refused). Returns the number of types enforced."""
    from . import primitives
    primitives.register_owner_events()
    n = 0
    for name, t in sorted(_TYPES.items()):
        if name in FORBIDDEN or not (t["owner"] == "kernel" or t["owner"] in world.modules):
            continue
        subscribe(world, name, _validator(name), owner="kernel")
        n += 1
    return n


def coverage(world, events=None) -> dict:
    """Which emitted event types of a world are typed: {emitted, typed, untyped: [...]} over `events` (default the
    in-memory log). Kernel bookkeeping types (FORBIDDEN, note, time, ...) are not counted."""
    kernel = FORBIDDEN | {"note", "time", "principal.update", "know", "proc.failed"}
    seen = sorted({e.type for e in (world.log if events is None else events)} - kernel)
    typed = [t for t in seen if t in _TYPES]
    return {"emitted": len(seen), "typed": len(typed), "untyped": [t for t in seen if t not in _TYPES]}


_VALIDATORS: dict[str, Callable] = {}


def _validator(name: str) -> Callable:
    fn = _VALIDATORS.get(name)
    if fn is None:
        def fn(ctx, ev, _name=name):
            errs = validate(_name, ev.payload)
            if errs:
                raise ProtocolError(f"invalid {_name} payload (seq {ev.seq}): {'; '.join(errs[:5])}")
        _VALIDATORS[name] = fn
    return fn
