"""World families that host a hidden mechanism as a facility.

For every registered mechanism family `F` (any engine; see `crucible.mechanisms`) two world families exist:

* `mech_F`         modules `identity`, `bank`, `admin`, `rig`: one facility `r1` hosting a mechanism of family F,
                   seat 0 pre-assigned to the handle `agent` (it gets the seat when it is admitted with
                   `admin.join`), further seats open to `rig.take`;
* `mech_F_tools`   the same plus `code`, `cloud`, `wf` and a compute provider (`mechinfra`), so a participant can
                   write a controller program, deploy it as a service, grant it `rig.operate` and trigger it every
                   tick: the program then reads gauges and sets levers through the same brokered tools.

The world family carries the mechanism family's split (train / val / test). The mechanism is drawn from the world
seed, so `(family, seed)` fixes the whole task. Overrides (`generate({"family": ..., "overrides": {...}}, seed)`)::

    mech                    use this mechanism (the output of make_mechanism) instead of drawing one
    mech_seed               draw make_mechanism(F, mech_seed) (default: a seed derived from the world seed)
    mech_overrides          passed to make_mechanism (e.g. {"seats": 2})
    validate, max_tries     draw with validity.sample_valid (slow: seconds per draw) and fail if nothing passes
    holders                 handles pre-assigned to seats 0, 1, ... (default ["agent"]; null leaves a seat open)
    tick_s                  tick length in seconds (default 3600)
    income_cents_per_tick   demand income (default: what the mechanism declares)
    lease_cents, history    as in the `rig` module config

`register_mech_world_families()` is idempotent: call it again after more engines have been imported.
"""
from __future__ import annotations

from .. import services  # noqa: F401  (registers identity, bank, admin, code, cloud, wf)
from ..core.kernel import Module, WorldSpec, register_module
from ..mechanisms import module as _rig  # noqa: F401  (registers the `rig` module)
from ..mechanisms.base import ENGINES, MECH_FAMILIES, make_mechanism
from ..mechanisms.module import rig_entry
from ..mechanisms.sim import MechSim, ensure_engines
from ..services.identity import ident
from .generator import FAMILIES, family

MECH_MODULES = ["identity", "bank", "admin", "rig"]
MECH_TOOL_MODULES = ["identity", "bank", "admin", "code", "cloud", "wf", "mechinfra", "rig"]
PROVIDER = {"handle": "cloud1", "name": "Cloud One", "cores": 16, "price": 20, "reliability": 1.0,
            "cash": 5_000_000}


@register_module
class MechInfra(Module):
    """The least a `mech_*_tools` world needs to run participants' programs: compute providers (organisations with
    a bank account and finite capacity that rent instances), without the rest of the digital economy."""
    name = "mechinfra"
    depends = ("identity", "bank", "cloud")

    def genesis(self, ctx):
        w = ctx.world
        idm, bank, cloud = ident(w), w.modules["bank"], w.modules["cloud"]
        for p in self.config.get("providers", []):
            owner = idm.create_person(ctx, f"{p['handle']}-ops", f"{p['name']} operations", "npc", {"role": "owner"})
            bank.open_account(ctx, owner, "checking")
            oid = idm.create_org(ctx, p["handle"], p["name"], owner, "company", {})
            acct = bank.open_account(ctx, oid, "operating")
            if int(p.get("cash", 0)) > 0:
                bank.move(ctx, "acct_genesis", acct, int(p["cash"]), "endowment", f"initial capital of {oid}")
            cloud.add_provider(ctx, oid, p["name"], int(p["price"]), int(p["cores"]), float(p["reliability"]))


def _builder(mech_family: str, tools: bool):
    def build(rng, o):
        mech = o.get("mech")
        if mech is None:
            mseed = int(o["mech_seed"]) if "mech_seed" in o else int(rng.integers(0, 2**31 - 1))
            if o.get("validate"):
                from ..mechanisms.validity import sample_valid       # operator side; only on request
                got = sample_valid(mech_family, mseed, int(o.get("max_tries", 8)), o.get("mech_overrides"))
                if got["mech"] is None:
                    raise ValueError(f"no valid {mech_family} mechanism in {got['tries']} draws: {got['rejected']}")
                mech = got["mech"]
            else:
                mech = make_mechanism(mech_family, mseed, o.get("mech_overrides"))
        n_seats = int(mech["config"].get("n_seats", 1))
        extra = {"history": int(o["history"])} if "history" in o else {}
        entry = rig_entry(mech, "r1", int(o.get("tick_s", 3600)), o.get("income_cents_per_tick"),
                          list(o.get("holders", ["agent"]))[:n_seats], int(o.get("lease_cents", 0)), **extra)
        modules = {"rig": {"rigs": [entry]}}
        params: dict = {}
        if tools:
            modules["mechinfra"] = {"providers": [dict(PROVIDER, **o.get("provider", {}))]}
            params = {"cost_model": "deterministic", "max_actions_per_hour": 240}
        return {"modules": modules, "params": params,
                "module_list": list(MECH_TOOL_MODULES if tools else MECH_MODULES)}
    return build


def register_mech_world_families() -> list[str]:
    """Register `mech_F` and `mech_F_tools` for every mechanism family whose engine is importable now. Safe to
    call any number of times; returns the names of all mechanism world families registered so far."""
    ensure_engines()
    for name in sorted(MECH_FAMILIES):
        fam = MECH_FAMILIES[name]
        if fam.engine not in ENGINES:
            continue
        for suffix, tools in (("", False), ("_tools", True)):
            wname = f"mech_{name}{suffix}"
            if wname in FAMILIES:
                continue
            doc = (f"A facility with a hidden mechanism of family {name} ({fam.engine} engine; transfer: "
                   f"{fam.transfer}). {fam.doc}" + (" With code, cloud and workflow services, so a deployed "
                                                    "controller program can operate the facility." if tools else ""))
            family(wname, fam.split, doc)(_builder(name, tools))
            FAMILIES[wname].economic = False     # no need-format economy: need-based goal families do not apply
    return mech_world_families()


def mech_world_families(split: str | None = None, tools: bool | None = None) -> list[str]:
    out = []
    for name in sorted(MECH_FAMILIES):
        for suffix, t in (("", False), ("_tools", True)):
            w = f"mech_{name}{suffix}"
            if w in FAMILIES and (split is None or FAMILIES[w].split == split) and (tools is None or tools == t):
                out.append(w)
    return out


def mech_of(spec: WorldSpec, rig_id: str = "r1") -> dict:
    """The hidden mechanism hosted by a world spec. Operator side only."""
    for m in spec.modules:
        if m["name"] == "rig":
            for r in m.get("config", {}).get("rigs", []):
                if r["id"] == rig_id:
                    return r["mech"]
    raise KeyError(f"no facility {rig_id!r} in this world spec")


def sim_of(spec: WorldSpec, rig_id: str = "r1", funds: list | None = None, money: bool = True) -> MechSim:
    """The kernel-free twin of a world's facility: same mechanism, same per-tick randomness, and (with `money`)
    the same demand income. `funds` are the holders' starting balances in cents (None: unlimited)."""
    for m in spec.modules:
        if m["name"] == "rig":
            for r in m.get("config", {}).get("rigs", []):
                if r["id"] == rig_id:
                    n = int(r["mech"]["config"].get("n_seats", 1))
                    held = [bool(s.get("holder")) for s in r.get("seats", [])] + [False] * n
                    return MechSim(r["mech"], spec.seed, rig_id,
                                   income_cents_per_tick=r["income_cents_per_tick"] if money else None,
                                   funds=funds, held=held[:n] if money else None)
    raise KeyError(f"no facility {rig_id!r} in this world spec")


register_mech_world_families()
