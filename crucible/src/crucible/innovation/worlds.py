"""World construction for the innovation and software-ecosystem mechanisms.

`innovation_spec(seed, ...)` composes a lean kernel world: identity, bank, operator tools, code hosting, search,
forum, issues, cloud, workflows, the v5 capability registry (`caps` with `registry: true`), the needs graph
(`needs`), packages (`pkg`), need-driven participants (`adopt`) and `infra`, which creates compute providers at
genesis. Tier 2 / Tier 3 generators (workstreams I and J) compose the same modules into their own worlds.

`FORMAT_SCHEMAS` are the JSON schemas of the public problem formats (crucible.economy.needs) as an operator would
declare them; `STANDARD_INTERFACES` are interface definitions a world may publish without any producer (used by the
needs graph's `complements` relation).
"""
from __future__ import annotations

from ..core.kernel import Module, WorldSpec, register_module

FORMAT_SCHEMAS = {
    "route.shortest": {
        "inputs": {"type": "object", "properties": {"n": {"type": "integer"}, "edges": {"type": "array"},
                                                    "source": {"type": "integer"}, "target": {"type": "integer"}},
                   "required": ["n", "edges", "source", "target"], "additionalProperties": True},
        "outputs": {"type": "object", "properties": {"path": {"type": "array", "items": {"type": "integer"}}},
                    "required": ["path"], "additionalProperties": True}},
    "sort.records": {
        "inputs": {"type": "object", "properties": {"records": {"type": "array"}, "descending": {"type": "boolean"}},
                   "required": ["records", "descending"], "additionalProperties": True},
        "outputs": {"type": "object", "properties": {"ids": {"type": "array"}}, "required": ["ids"],
                    "additionalProperties": True}},
    "text.extract": {
        "inputs": {"type": "object", "properties": {"documents": {"type": "array"}}, "required": ["documents"],
                   "additionalProperties": True},
        "outputs": {"type": "object", "properties": {"records": {"type": "array"}}, "required": ["records"],
                    "additionalProperties": True}},
    "schedule.jobs": {
        "inputs": {"type": "object", "properties": {"durations": {"type": "array"}, "machines": {"type": "integer"}},
                   "required": ["durations", "machines"], "additionalProperties": True},
        "outputs": {"type": "object", "properties": {"assignment": {"type": "array"}}, "required": ["assignment"],
                    "additionalProperties": True}},
    "pack.knapsack": {
        "inputs": {"type": "object", "properties": {"items": {"type": "array"}, "capacity": {"type": "integer"}},
                   "required": ["items", "capacity"], "additionalProperties": True},
        "outputs": {"type": "object", "properties": {"chosen": {"type": "array"}}, "required": ["chosen"],
                    "additionalProperties": True}},
}

STANDARD_INTERFACES = [
    {"name": "route.render", "doc": "Render a route (a list of node ids) as a turn-by-turn itinerary.",
     "inputs": {"type": "object", "properties": {"path": {"type": "array"}}, "required": ["path"]},
     "outputs": {"type": "object", "properties": {"steps": {"type": "array"}}, "required": ["steps"]}},
]


@register_module
class Infra(Module):
    """Compute providers created at genesis (config `providers`: [{handle, name, cores, price, reliability,
    cash}]). Providers are ordinary organisations with bank accounts; the cloud module meters and bills."""
    name = "infra"
    depends = ("identity", "bank", "cloud")

    def genesis(self, ctx):
        idm, bank, cloud = (ctx.world.modules[m] for m in ("identity", "bank", "cloud"))
        for p in self.config.get("providers") or [{"handle": "cloudco", "name": "CloudCo", "cores": 64, "price": 100,
                                                    "reliability": 1.0, "cash": 5_000_000}]:
            owner = idm.create_person(ctx, f"{p['handle']}-ceo", f"{p['name']} CEO", "npc", {"role": "owner"})
            bank.open_account(ctx, owner, "checking")
            oid = idm.create_org(ctx, p["handle"], p["name"], owner, "company", {})
            acct = bank.open_account(ctx, oid, "operating")
            if p.get("cash"):
                bank.move(ctx, "acct_genesis", acct, int(p["cash"]), "endowment", f"initial capital of {oid}")
            cloud.add_provider(ctx, oid, p["name"], int(p["price"]), int(p["cores"]), float(p.get("reliability", 1.0)))


BASE = ["identity", "bank", "admin", "code", "search", "forum", "issues", "cloud", "wf"]


def innovation_spec(seed: int, *, participants: int = 24, needs: dict | None = None, channels: dict | None = None,
                    adopt: dict | None = None, needs_cfg: dict | None = None, providers: list | None = None,
                    params: dict | None = None, with_adopters: bool = True, family: str = "innovation") -> WorldSpec:
    """A world with the v5 registry, the needs graph, packages and need-driven participants."""
    mods = [{"name": n, "config": {}} for n in BASE]
    mods[BASE.index("forum")]["config"] = {"boards": ["general", "dev", "market"]}
    mods.append({"name": "caps", "config": {"registry": True}})
    mods.append({"name": "infra", "config": {"providers": providers} if providers else {}})
    mods.append({"name": "pkg", "config": {}})
    mods.append({"name": "needs", "config": {"interfaces": STANDARD_INTERFACES, **(needs_cfg or {})}})
    if with_adopters:
        cfg = {"n": participants, "needs": needs or {"route.shortest": 1.0},
               "channels": channels or {"search": 1.0}, **(adopt or {})}
        mods.append({"name": "adopt", "config": cfg})
    p = {"cost_model": "deterministic", "cu_per_ms": 50.0}
    p.update(params or {})
    return WorldSpec(family, int(seed), mods, p)
