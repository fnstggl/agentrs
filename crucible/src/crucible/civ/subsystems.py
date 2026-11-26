"""The subsystems of a Tier 3 civilisation (MASTER_SPEC Part 2.1 Mode C) and the kernel modules that implement them.

Mode C names seventeen major domains that must share one world state. Each entry below lists the modules that carry
the domain in a v5 civilisation, the shared facts the domain reads or writes (identities, accounts, artifacts,
resources), and how the one population takes part in it. A module belongs to exactly one subsystem for counting.

The population (`crucible.population`, D-15) is not a subsystem of its own: it is the source of every simulated
human decision in all of them.

Some modules exist in the tree but are excluded from every civilisation because they keep a second population of
simulated people or a scheduled narrative (ONE_WORLD_CONTRACTS section 1, D-14/D-15); `EXCLUDED` gives the reason.

`readiness()` states which subsystem modules cannot yet be composed with the population in this checkout and what
they wait for (a module whose population hook is not exported yet). The generator leaves those out of the world and
records them as pending; the gate tests count a subsystem only when it is present AND active (see
`crucible.civ.integration`), so a pending subsystem fails the gate instead of being assumed.
"""
from __future__ import annotations

import importlib

# --------------------------------------------------------------------------------------------- Mode C subsystems
SUBSYSTEMS: dict[str, dict] = {
    "digital_economy": {
        "modules": ["business", "products", "subs", "ventures", "popsites"],
        "doc": "companies, digital products, plans and subscriptions, web shops the population buys from",
        "population": "web visits and purchases through each site's own checkout (popweb); product/plan purchases "
                      "through the commerce activities (ECON)"},
    "social_media": {
        "modules": ["content", "social_ids", "social_rows", "social", "aud", "feed", "creators", "msg", "studio"],
        "doc": "content catalogue, follow/share graph, exact audience accounting, recommender feeds, creators, "
               "messages",
        "population": "feed browsing (recsys `browse` activity) and posting/sharing (CORE-POP)"},
    "search_information": {
        "modules": ["search", "forum", "kb", "docs"],
        "doc": "search index, discussion boards, knowledge base, documents",
        "population": "search and forum activities (popweb)"},
    "scientific_research": {
        "modules": ["sci"],
        "doc": "phenomena with hidden laws, apparatus, experiments, papers, replication",
        "population": "research activity (wave 2b); autonomous labs"},
    "software_development": {
        "modules": ["code", "issues", "pkg", "wf"],
        "doc": "repositories, issues, packages with dependency edges, workflows",
        "population": "developer activities (ECON)"},
    "technology_invention": {
        "modules": ["caps", "needs"],
        "doc": "typed capability declarations, trials and adoption; needs from activity bottlenecks",
        "population": "trying services on own activity inputs (ECON)"},
    "industry": {
        "modules": ["ops", "industry", "society", "industrygen"],
        "doc": "materials, machines, process designs and production runs; firms run by organisational minds",
        "population": "households buy physical goods (`goods` activity, crucible.civ.goods)"},
    "supply_chains": {
        "modules": ["freight"],
        "doc": "freight network between regions; standing orders and offers between firms (industry)",
        "population": "availability and prices of goods the population buys"},
    "energy": {
        "modules": [],
        "doc": "regional grids (state of `industry`: make.grid, make.energy) and the datacenter energy draw of "
               "compute (coupling `compute_energy`); counted from its events and grid state (EVENT_SUBSYSTEM)",
        "population": "indirect: prices of goods and services"},
    "computing_infrastructure": {
        "modules": ["cloud", "net", "appdb", "fleet"],
        "doc": "compute providers, instances, deployments, metering, network, datacenter fleet operation",
        "population": "every page view of a deployed site is a metered request"},
    "employment": {
        "modules": ["labor", "jobs", "firms"],
        "doc": "labour market, job postings, organisations that recruit and run projects",
        "population": "employment and job search (CORE-POP/ECON: pop.employer -> labor)"},
    "education": {
        "modules": ["learn"],
        "doc": "courses, concepts, assessments",
        "population": "study activity (Tier 2 education, wave 2b)"},
    "finance": {
        "modules": ["bank", "contracts", "commit", "deal"],
        "doc": "one bank (accounts, ledger, invoices, mandates), contracts, commitments, deals",
        "population": "every payment of an individual is a bank move from its own account"},
    "communities": {
        "modules": ["community"],
        "doc": "communities with membership, roles and moderation",
        "population": "joining and taking part (CORE-POP)"},
    "organizations": {
        "modules": ["identity", "civgen"],
        "doc": "one identity registry: persons, organisations, membership, grants; civilisation genesis",
        "population": "individuals are principals p<i> of the same registry"},
    "institutions": {
        "modules": ["inst", "gov"],
        "doc": "institutional charters and governance constitutions",
        "population": "members vote and are bound by rules"},
    "marketplaces": {
        "modules": ["xchg", "market"],
        "doc": "listing marketplaces (v5 exchange, 4.0 market)",
        "population": "buying listings (ECON)"},
}

# the population and its activity modules (people's own decisions: browsing, creating, working, buying goods)
POPULATION_MODULES = ["pop", "popweb", "pop_adaptive", "pop_strategic", "pop_social", "work", "civgoods"]
CIV_MODULES = ["civlinks"]          # the coupling engine (not a domain of its own)
SUBSTRATE_MODULES = ["admin", "artifacts", "cols", "assets", "space"]   # space: the one shared semantic space (D-16)

# Modules that keep a second population of simulated people (or scripted narrative). Never part of a civilisation.
EXCLUDED: dict[str, str] = {
    "crowd": "social stand-in population (C2); replaced by crucible.population (D-15)",
    "patrons": "commerce customer NPCs with fixed needs (F1); replaced by the population (D-15)",
    "adopt": "innovation adopters with predefined formats and hidden-truth verifiers (G); replaced by the population",
    "consumers": "4.0 pooled-household consumers of need formats; replaced by the population",
    "shoppers": "4.0 household goods demand; replaced by the population's `goods` activity (crucible.civ.goods)",
    "npcs": "4.0 NPC persons (workers, founders); people come from the population",
    "civ": "4.0 digital-civilisation genesis (founders are NPC persons; depends on `npcs`)",
    "shocks": "scheduled exogenous shocks (scripted narrative; Part 10.3)",
    "calendar": "alias of `cal` (duplicate appliers)",
    "use": "4.0 usefulness lab: its tools return hidden-evaluator quality scores (audit G S5)",
    "mechinfra": "4.0 mechanism-family infrastructure (Tier 1)",
    "rig": "hidden-mechanism facilities (Tier 1)",
    "adversary": "red-team world module",
    "lab": "4.0 engineering lab family (standalone world)",
    "infra": "innovation scenario infrastructure (lean scenario worlds, audit G 5.5)",
    "seat": "Tier 2 domain seat module (domains framework)",
    "webgen": "4.0 generated-site family (popsites is the v5 equivalent)",
    "entry": "4.0 market-entry incumbents (need formats)",
}

MODULE_SUBSYSTEM: dict[str, str] = {m: s for s, d in SUBSYSTEMS.items() for m in d["modules"]}
for _m in POPULATION_MODULES:
    MODULE_SUBSYSTEM[_m] = "population"
for _m in SUBSTRATE_MODULES:
    MODULE_SUBSYSTEM[_m] = "substrate"
for _m in CIV_MODULES:
    MODULE_SUBSYSTEM[_m] = "couplings"
# auxiliary modules a civilisation may load (counted with the subsystem they serve)
MODULE_SUBSYSTEM.update({"mail": "search_information", "support": "digital_economy", "cal": "organizations",
                         "ads": "digital_economy", "goals": "organizations", "exp": "organizations",
                         "embed": "social_media", "media": "social_media", "letters": "social_media",
                         "launch": "social_media", "dev": "search_information", "tube": "social_media",
                         "shorts": "social_media", "posts": "social_media", "blog": "social_media",
                         "rivals": "digital_economy",
                         "pics": "social_media", "pods": "social_media", "semantic_inference": "social_media"})

# event types that belong to another subsystem than their module's (the regional grids live in `industry` but are
# the energy system; trade between firms is the supply chain)
EVENT_SUBSYSTEM: dict[str, str] = {
    "make.grid": "energy", "make.grid_order": "energy", "make.grid_done": "energy", "make.energy": "energy",
    "links.energy": "energy", "make.offer": "supply_chains", "make.order": "supply_chains",
    "make.trade": "supply_chains", "make.standing": "supply_chains", "make.standing_status": "supply_chains",
    "make.dispatch": "supply_chains", "make.retail": "supply_chains", "make.retail_update": "supply_chains",
    "make.retail_status": "supply_chains",
    # coupling events belong to the subsystem whose state they describe
    "links.rnd": "industry", "links.rnd_result": "industry", "links.curtailed": "computing_infrastructure",
    "links.backup": "energy", "links.coordination": "organizations", "goods.wom": "social_media",
}


def host_module_map(world) -> dict[str, str]:
    """event-type prefix (host or module name) -> module name, for a world."""
    out = {}
    for name, mod in world.modules.items():
        out[name] = name
        out[mod.host or name] = name
    # 4.0 hosts whose event prefixes are not the module host
    for prefix, mod in (("id", "identity"), ("db", "appdb"), ("make", "industry"), ("biz", "business"),
                        ("research", "consumers"), ("venture", "ventures"), ("prod", "products"),
                        ("artifact", "artifacts"), ("asset", "assets")):
        if mod in world.modules:
            out.setdefault(prefix, mod)
    return out


def subsystem_of_module(name: str) -> str:
    return MODULE_SUBSYSTEM.get(name, "other")


def subsystem_of_event(world, etype: str, hm: dict | None = None) -> str | None:
    """Subsystem whose module emitted an event type (by type prefix); None for kernel bookkeeping."""
    prefix = etype.split(".", 1)[0]
    if prefix in ("cmd", "proc", "sched", "time", "ext", "idem", "ctr", "know", "note", "principal"):
        return None
    if etype in EVENT_SUBSYSTEM:
        return EVENT_SUBSYSTEM[etype]
    hm = hm or host_module_map(world)
    mod = hm.get(prefix)
    return subsystem_of_module(mod) if mod else None


# --------------------------------------------------------------------------------------------- readiness
def _population_exports(name: str) -> bool:
    try:
        pkg = importlib.import_module("crucible.population")
    except ImportError:
        return False
    return callable(getattr(pkg, name, None))


# module -> (predicate, what it waits for). A module listed here is composed only when its predicate holds.
REQUIREMENTS: dict[str, tuple] = {
    "feed": (lambda: _population_exports("register_activity"),
             "crucible.population must export register_activity (CORE-POP, ONE_WORLD_CONTRACTS section 1); "
             "social/recsys/service.py refuses to install without it"),
}


def readiness() -> dict[str, dict]:
    """{module: {ready, waits_for}} for every module with an integration precondition."""
    out = {}
    for m, (pred, why) in REQUIREMENTS.items():
        try:
            ok = bool(pred())
        except Exception:  # noqa: BLE001 - a failing probe means not ready
            ok = False
        out[m] = {"ready": ok, "waits_for": "" if ok else why}
    return out
