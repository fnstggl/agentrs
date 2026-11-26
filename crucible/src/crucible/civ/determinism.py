"""Determinism checks of civilisations under contract 1.7 (crucible.substrate.inference; MASTER_SPEC Part 10.4:
"deterministic recorded replay", "independent replication", snapshots, resume and forks).

A civilisation's world logic is deterministic; batched neural inference on a shared GPU (and measured sandbox CPU
time) is not bit-reproducible, which is why the kernel records those outcomes in `ext` events. Two runs of one spec
and seed that recompute inference can therefore part ways when one inference outcome differs in its last bits and a
decision downstream flips. These helpers tell the two apart and check the strong property directly:

    first_divergence(a, b)          where two event logs over the same seq range part and why: the first event whose
                                    hash differs, its kernel transaction, and whether that transaction's recorded
                                    external outcomes differ ("recorded_outcome") or not ("logic": the world's own
                                    logic, a fork or a resume left a trace)
    regenerate_continuation(w, evs) re-execute the recorded commands, process runs and clock advances `evs` that
                                    followed `w`'s state through the handlers with their recorded external outcomes
                                    injected - `World.regenerate` started from a loaded snapshot instead of genesis;
                                    if snapshot, load and world logic are exact, the result equals the recorded world

Events may be kernel `Event`s or their dict form (`Event.to_dict`, store rows).
"""
from __future__ import annotations

import json

BOUNDARY_EVENTS = ("cmd", "cmd.rejected", "proc", "time")    # the first event of a kernel transaction


def _g(e, k):
    return e[k] if isinstance(e, dict) else getattr(e, k)


def first_divergence(a: list, b: list) -> dict:
    n = min(len(a), len(b))
    i0 = next((i for i in range(n) if _g(a[i], "hash") != _g(b[i], "hash")), None)
    if i0 is None:
        same = len(a) == len(b)
        return {"equal": same, "first_seq": None, "origin": None if same else "length",
                "len_a": len(a), "len_b": len(b)}

    def txn(evs, i):
        j = i
        while j > 0 and _g(evs[j], "type") not in BOUNDARY_EVENTS:
            j -= 1
        k = j + 1
        while k < len(evs) and _g(evs[k], "type") not in BOUNDARY_EVENTS:
            if _g(evs[k], "type") == "ext":
                return _g(evs[j], "type"), _g(evs[k], "payload")
            k += 1
        return _g(evs[j], "type"), None
    ta, xa = txn(a, i0)
    tb, xb = txn(b, i0)
    differ = xa is not None and xb is not None and json.dumps(xa, sort_keys=True, default=str) != \
        json.dumps(xb, sort_keys=True, default=str)
    kinds = sorted({str(o.get("kind", "?")) for x in (xa, xb) if x for o in (x.get("outcomes") or [])
                    if isinstance(o, dict)})
    return {"equal": False, "first_seq": int(_g(a[i0], "seq")), "type_a": _g(a[i0], "type"),
            "type_b": _g(b[i0], "type"), "transaction": ta, "ext_kinds": kinds,
            "origin": "recorded_outcome" if differ else "logic"}


def regenerate_continuation(w, events: list) -> None:
    """Mirror of the loop of `crucible.core.kernel.World.regenerate` on an already loaded world `w` (its state is the
    one the first event of `events` followed). Uses the kernel's regeneration hook `World._regen` (the recorded
    external outcomes by transaction key), exactly as `World.regenerate` does."""
    from ..core.errors import InvalidAction
    from ..core.kernel import Event, _MALFORMED
    evs = [e if not isinstance(e, dict) else Event.from_dict(e) for e in events]
    ext_by_cmd: dict = {}
    for e in evs:
        if e.type == "ext":
            k = e.payload.get("key", e.cmd)
            if k is not None:
                ext_by_cmd.setdefault(k, []).extend(e.payload["outcomes"])
        if e.type in ("cmd.rejected", "proc.failed") and e.payload.get("ext"):
            k = e.payload.get("ext_key", e.cmd)
            if k is not None:
                ext_by_cmd.setdefault(k, []).extend(e.payload["ext"])
    w._regen = ext_by_cmd
    try:
        for j, e in enumerate(evs):
            if e.seq <= w.seq:
                continue
            if e.type == "cmd" and e.payload.get("n") is not None:
                w.execute(e.actor, {"tool": e.payload["tool"], "args": e.payload["args"],
                                    **({"idempotency_key": e.payload["key"]} if e.payload.get("key") is not None
                                       else {})})
            elif e.type == "cmd" and e.payload.get("system"):
                w.system_call(e.payload["tool"], e.payload["args"], as_principal=e.payload["as"])
            elif e.type == "cmd.rejected":
                actor = e.payload.get("actor") or e.actor
                if e.payload.get("n") is None and e.payload["error"].get("message") == _MALFORMED:
                    w._reject(actor, e.payload["tool"], e.payload["args"], InvalidAction(_MALFORMED))
                    continue
                act = {"tool": e.payload["tool"], "args": e.payload["args"]}
                nxt = evs[j + 1] if j + 1 < len(evs) else None
                if e.cmd is not None and nxt is not None and nxt.type == "idem" and nxt.cmd == e.cmd:
                    act["idempotency_key"] = nxt.payload["id"][len(actor) + 1:]
                w.execute(actor, act)
            elif e.type == "proc":
                w.advance_time(e.payload["at"], max_jobs=1)
            elif e.type == "time":
                w.advance_time(e.payload["to"])
    finally:
        w._regen = None
