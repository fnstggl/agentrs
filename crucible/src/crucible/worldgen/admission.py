"""Admission tests for new world modules (human-approved world evolution).

A new application or mechanism (hand-written or generated) is a kernel module. Before it may be added to a world
family it must pass `admit(module_names, ...)`, which builds a world containing it and drives a randomised session
of schema-valid tool calls from several principals (plus time advancing), then checks:

1. every endpoint has a valid input schema (`schema.check_schema`) and a description;
2. no handler crashed (an internal error is a module bug, not an agent outcome);
3. `check_invariants()` is empty, money is conserved (sum of balances 0) and the ledger is consistent;
4. replay from the event log reproduces the state digest exactly;
5. regeneration from the recorded commands reproduces the event log exactly (same hash chain);
6. snapshot -> load round trip and fork isolation;
7. visibility: no principal can see another principal's private events (vis tuples), checked over the log;
8. the module does not register hooks that replace kernel-protected hooks (`principals_of`, `may_act_as`, `bank`).

Agent-created software never goes through this path: it runs inside the synthetic internet as sandboxed services.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core import schema as S
from ..core.kernel import MODULES, World, WorldSpec

PROTECTED_HOOKS = ("principals_of", "may_act_as", "bank")


@dataclass
class AdmissionReport:
    module: str
    passed: bool
    checks: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)


def admit(new_modules: list[str], base: list[str] | None = None, steps: int = 400, actors: int = 4, seed: int = 0,
          configs: dict | None = None) -> AdmissionReport:
    from ..agents.scripted import _random_args
    base = base or ["identity", "bank", "admin"]
    mods = [{"name": m, "config": (configs or {}).get(m, {})} for m in base + [m for m in new_modules if m not in base]]
    spec = WorldSpec("admission", seed, mods, {})
    rep = AdmissionReport(",".join(new_modules), False)
    try:
        w = World.create(spec)
    except Exception as e:  # noqa: BLE001
        rep.errors.append(f"world creation failed: {type(e).__name__}: {e}")
        return rep
    hooks_before = {k: w.hooks.get(k) for k in PROTECTED_HOOKS}
    # 1. schemas
    bad = []
    for name, ep in sorted(w.endpoints.items()):
        if ep.host not in [MODULES[m].host or m for m in new_modules if m in MODULES]:
            continue
        errs = S.check_schema(ep.params)
        if errs or not ep.doc:
            bad.append((name, errs or ["missing description"]))
    rep.checks["schemas"] = not bad
    rep.errors += [f"schema {n}: {e}" for n, e in bad]
    # randomised session
    rng = np.random.default_rng(seed)
    principals = []
    for i in range(actors):
        principals.append(w.system_call("admin.join", {"handle": f"adm{i}", "endowment": 1_000_000})["principal"])
    own = [e for e in w.endpoints.values() if e.host != "admin"]
    crashes = 0
    for step in range(steps):
        p = principals[int(rng.integers(0, len(principals)))]
        ep = own[int(rng.integers(0, len(own)))]
        args = _random_args(ep.params, rng, p)
        r = w.execute(p, {"tool": ep.name, "args": args})
        if not r.ok and str(r.error.get("message", "")).startswith("internal error"):
            crashes += 1
            rep.errors.append(f"crash in {ep.name}: {r.error['message'][:200]}")
        if step % 50 == 49:
            w.advance_time(w.now + 3600 * int(rng.integers(1, 12)))
    rep.checks["no_crashes"] = crashes == 0
    inv = w.check_invariants()
    bank = w.modules.get("bank")
    total = sum(a["balance"] for a in w.state.tables["accounts"].values()) if bank else 0
    ledger = bank.check_ledger_consistency(w) if bank else []
    rep.checks["invariants"] = not inv and total == 0 and not ledger
    rep.errors += inv + ledger + ([f"money not conserved: {total}"] if total else [])
    evs = [e.to_dict() for e in w.log]
    try:
        rep.checks["replay"] = World.replay(spec, evs).digest() == w.digest()
    except Exception as e:  # noqa: BLE001
        rep.checks["replay"] = False
        rep.errors.append(f"replay: {e}")
    try:
        g = World.regenerate(spec, evs)
        rep.checks["regeneration"] = g.last_hash == w.last_hash
        if not rep.checks["regeneration"]:
            for a, b in zip(w.log, g.log):
                if a.hash != b.hash:
                    rep.errors.append(f"regeneration diverged at seq {a.seq}: {a.type} vs {b.type}")
                    break
    except Exception as e:  # noqa: BLE001
        rep.checks["regeneration"] = False
        rep.errors.append(f"regeneration: {e}")
    try:
        blob = w.snapshot_blob()
        r2 = World.from_blob(blob)
        f = w.fork("adm-fork")
        d0 = w.digest()
        f.execute(principals[0], {"tool": "world.clock", "args": {}})
        rep.checks["snapshot_fork"] = r2.digest() == d0 and w.digest() == d0 and f.digest() != d0
    except Exception as e:  # noqa: BLE001
        rep.checks["snapshot_fork"] = False
        rep.errors.append(f"snapshot/fork: {e}")
    leaks = 0
    for e in w.log:
        if isinstance(e.vis, tuple):
            for p in principals:
                if p not in e.vis and w.visible(e, w.principals_of(p)) and not (w.principals_of(p) & set(e.vis)):
                    leaks += 1
    rep.checks["visibility"] = leaks == 0
    rep.checks["protected_hooks"] = all(w.hooks.get(k) is hooks_before[k] for k in PROTECTED_HOOKS)
    rep.passed = all(rep.checks.values())
    return rep
