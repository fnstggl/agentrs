# World kernel

Code: `crucible/src/crucible/core/` (`kernel.py`, `state.py`, `schema.py`, `canonical.py`, `errors.py`) and
`crucible/src/crucible/storage/`. Tests: `crucible/tests/test_kernel*.py`, `test_storage.py`.

## 1. Model

A Crucible world is an **event-sourced state machine** with one authoritative state:

```
command (actor, tool, args) ──handler──▶ events ──appliers──▶ state      (one transaction)
scheduled process (t, name)  ──handler──▶ events ──appliers──▶ state      (one transaction)
```

* **Handlers** (endpoint functions and scheduled processes) validate, read state, draw seeded randomness and
  *emit events*. They never write state directly.
* **Appliers** are deterministic functions `(state, event) → state'`. Replaying the event log on an empty world
  reproduces the state bit-for-bit (`World.replay`), and the event hash chain (`blake2b(prev || canonical(event))`)
  makes edits to history detectable.
* **Transactions.** A command (or process run) applies its events as it emits them, journaling every overwritten
  record; any `CrucibleError` rolls the whole command back (`StateStore.rollback`). Nested service calls use
  savepoints (`Ctx.call`). Rejections are recorded as a `cmd.rejected` event so the learner and the audit see them.
* **Regeneration.** Commands and clock advances are themselves recorded (`cmd`, `time` events), so
  `World.regenerate` re-executes the *handlers* and must produce the identical event log (same hashes). This is the
  determinism test for the kernel and every module. Outcomes the kernel cannot recompute (sandboxed program output,
  measured CPU time) go through `ctx.external(...)`; they are recorded once and injected during regeneration.
* **Time.** Integer simulated seconds in `state.meta["clock"]`. It moves only via `advance_time(until)`, which runs
  due jobs in `(time, priority, sequence)` order. Jobs live in the `sched` table (so they are part of snapshots).

## 2. State store

`StateStore` holds named tables of JSON-like records (`dict`s) with declared secondary indexes, dense numpy
arrays (for background populations) and a small `meta` dict (clock, counters). `get`/`find` return read-only
views; iteration is in sorted id order (no dependence on hash seeds). Counters (`new_id`) are allocated by
handlers; the kernel records their final values in a `ctr` event at commit so replay reproduces them.

## 3. Writing a module

```python
from crucible.core import schema as S
from crucible.core.kernel import Module, endpoint, applier, process, register_module

@register_module
class Notes(Module):
    name = "notes"           # module name (WorldSpec refers to it)
    host = "notes"           # tool prefix and synthetic hostname (notes.crucible)
    depends = ("identity",)

    def install(self, world):                 # declare tables & indexes, register hooks (no state writes)
        world.state.declare_table("notes", index=("owner",))

    def genesis(self, ctx):                   # initial content: emit events
        ...

    @endpoint("write", S.obj({"text": S.s_str(maxLength=500)}, ["text"]), cost_s=5)
    def ep_write(self, ctx, text):
        """Docstring becomes the tool description."""
        nid = ctx.state.new_id("note")
        ctx.emit("notes.write", {"id": nid, "owner": ctx.actor, "text": text}, vis=(ctx.actor,))
        return {"note": nid}

    @applier("notes.write")
    def ap_write(self, ev):
        self.world.state.insert("notes", ev.payload["id"], {"owner": ev.payload["owner"], "text": ev.payload["text"], "t": ev.t})

    @process("cleanup")                       # scheduled as "notes.cleanup"
    def pr_cleanup(self, ctx, **args): ...

    def check_invariants(self, world): return []    # list of violated invariants (strings)
    def observe(self, world, actor): return None     # small per-actor summary added to observations
```

Rules every module must follow (checked by the regeneration and invariant tests):

1. Appliers read only `self.world.state` and the event. No randomness, no wall clock, no external I/O.
2. Handlers draw randomness only from `ctx.rng(...)` or `world.rng_for(...)` (seeded from the world seed and the
   event sequence number) and put every random outcome into an event payload.
3. Authorisation: before touching a resource another principal owns, call
   `ident(world).authorize(ctx, owner, "<cap>", amount=..., scope=...)` (see `services/identity.py`).
4. Money moves only through `bank.post/move/debit` (double-entry; the sum of all balances is always 0).
5. Visibility: `vis=(principals...)` for private events, `"public"` for world-visible ones, `"internal"` for
   audit-only ones. Observations never include events an actor cannot see.
6. Iterate tables in sorted order (`scan`, `find_ids`), never over Python sets.
7. Use `ctx.external(kind, fn)` for anything non-deterministic (program execution, timing measurements).

## 4. Principals, tools and discovery

Principals (`principals` table): persons `p:<handle>`, organisations `o:<handle>`, service principals of
deployed software `svc:<host>`, and `sys`. Built-in services are the well-known part of the internet: their tools
(`<host>.<endpoint>`, e.g. `bank.transfer`) are always listed. Agent-deployed services become tools for a
principal once that principal has *discovered* the host (search results, messages, listings, `world.describe`),
recorded in the `knowledge` table; `World.available_tools(principal)` is therefore dynamic.
`admin.*` tools exist so that operator setup and interventions are recorded commands; the kernel rejects them
for every in-world actor.

## 5. Persistence, snapshots, forks

* `Store` (memory / SQLite / PostgreSQL) holds the append-only event log per world (`(world_id, seq)` unique,
  appends must be contiguous), immutable snapshot blobs, and world lineage (`parent = {world, seq, hash}`).
* `World.snapshot_blob()` = zlib(pickle(spec, state, seq, hash)); `World.from_blob(..., fork=True)` creates an
  isolated world that shares history up to the snapshot (copy-on-snapshot). Forks never share mutable objects.
* Crash recovery: load the latest snapshot, replay events after it, verify the hash chain.
* Concurrency: one writer per world (`Store.acquire` lease); the contiguity check on append rejects a second
  writer even if leases are bypassed.

## 6. Reproducibility guarantees and limits

| Mechanism | Guarantee |
|---|---|
| Event replay | exact (digest-equal), verified by tests |
| Regeneration from commands | exact for kernel and built-in modules, with recorded external outcomes |
| Re-running agent programs | not guaranteed (timing, program nondeterminism); outcomes are recorded instead |
| Snapshots/forks | exact copies; divergence only through subsequent commands |
