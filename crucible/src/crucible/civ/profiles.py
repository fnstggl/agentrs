"""Civilisation profiles (MASTER_SPEC Part 10.1): structural variants of one Tier 3 civilisation.

Every profile enables every subsystem (Mode C, D-17). Profiles differ in *structure*, never only in starting cash:

* `couplings`   which cross-domain relations exist (crucible.civ.couplings; a profile without `goods_wom` has no
                word-of-mouth channel for physical goods, one without `science_process` no route from published
                science to industrial practice, ...);
* `industry`    the supply-chain graph (`chain`: which firms make what from whose output), the regions and the
                freight-link topology between them, the number of materials;
* `energy`      the grid regime (capacity relative to load) and whether utilities may expand;
* `compute`     the compute market (number of providers, capacity regime, where datacenters stand);
* `social`      how the population is placed in regions (`placement`: contiguous blocks keep each person's contacts
                inside one region - fragmented; scattered placement links regions - interconnected) and whether
                discussion boards are regional, topical or global;
* `institutions` the constitutions organisations adopt (equity, equal or labour-hours profit sharing; public
                ownership of utilities), which change effort and decisions through `gov`;
* `science`     how many phenomena exist, how many of them are the physics of an industrial process, how many
                rival and replication labs work on them;
* `digital`     the generated web (number of sites, share of shops, category mix).

`structural_key(profile)` is the discrete part of a profile (sets, graphs, categories); two profiles whose keys are
equal differ only in parameters. The gate tests measure the same thing on *built* worlds
(`crucible.civ.integration.fingerprint`), not on these declarations.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ..core.canonical import digest

ALL_COUPLINGS = ("science_process", "compute_energy", "grid_curtailment", "goods_supply", "goods_wom",
                 "org_coordination", "education_skills", "freight_energy", "lab_energy", "energy_cost_pricing",
                 "datacenter_siting")


@dataclass(frozen=True)
class Profile:
    name: str
    title: str
    doc: str
    couplings: tuple[str, ...]
    industry: dict = field(default_factory=dict)
    energy: dict = field(default_factory=dict)
    compute: dict = field(default_factory=dict)
    social: dict = field(default_factory=dict)
    institutions: dict = field(default_factory=dict)
    science: dict = field(default_factory=dict)
    digital: dict = field(default_factory=dict)
    part_10_1: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "title": self.title, "doc": self.doc, "couplings": list(self.couplings),
                "industry": copy.deepcopy(self.industry), "energy": dict(self.energy), "compute": dict(self.compute),
                "social": dict(self.social), "institutions": copy.deepcopy(self.institutions),
                "science": dict(self.science), "digital": copy.deepcopy(self.digital), "part_10_1": self.part_10_1}


BASE = dict(
    industry={"chain": "standard", "regions": 4, "links": "ring", "materials": 6},
    energy={"regime": "normal", "expand": True},
    compute={"market": "duopoly", "regime": "normal", "datacenters": "spread"},
    social={"placement": "mixed", "boards": "topical"},
    institutions={"governance": "equity", "utilities": "private"},
    science={"phenomena": 2, "bound": 1, "rivals": 2, "replicators": 1},
    digital={"sites": 20, "shop_share": 0.4},
)


def _p(name, title, doc, part, couplings, **over) -> Profile:
    d = copy.deepcopy(BASE)
    for k, v in over.items():
        d[k] = {**d[k], **v}
    return Profile(name, title, doc, tuple(couplings), part_10_1=part, **d)


_C = ALL_COUPLINGS
PROFILES: dict[str, Profile] = {p.name: p for p in [
    _p("balanced", "Balanced civilisation",
       "Reference profile: a four-region ring economy with the standard supply chain, a compute duopoly, topical "
       "boards and every coupling.", "", _C),
    _p("digital_first", "Digital-first",
       "A large generated web with many shops and a competitive compute market on a short, linear supply chain; "
       "datacenters are the largest energy load; science is small and not tied to industry.",
       "Digital-first", ("compute_energy", "grid_curtailment", "goods_supply", "goods_wom", "education_skills",
                         "energy_cost_pricing", "datacenter_siting"),
       industry={"chain": "linear", "regions": 3, "links": "line"},
       compute={"market": "competitive", "regime": "abundant", "datacenters": "spread"},
       digital={"sites": 32, "shop_share": 0.55},
       science={"phenomena": 1, "bound": 0, "rivals": 1, "replicators": 1}),
    _p("industrial_first", "Industrial-first",
       "A wide five-region supply chain (three mines, three component makers, four competing assemblers) with "
       "chords in the freight ring; a single compute provider; few web sites.",
       "Industrial-first", ("science_process", "compute_energy", "goods_supply", "goods_wom", "org_coordination",
                            "education_skills", "freight_energy", "lab_energy"),
       industry={"chain": "wide", "regions": 5, "links": "ring_chords", "materials": 7},
       compute={"market": "monopoly", "regime": "normal", "datacenters": "colocated"},
       digital={"sites": 10, "shop_share": 0.3}),
    _p("research_intensive", "Research-intensive",
       "Four phenomena, every one the physics of an industrial process, worked on by rival and replication labs; "
       "the R&D route from replicated science to process settings exists; a diamond supply chain (every assembler "
       "needs both component makers).",
       "Research-intensive", ("science_process", "compute_energy", "grid_curtailment", "goods_supply", "goods_wom",
                              "education_skills", "lab_energy", "energy_cost_pricing"),
       industry={"chain": "diamond"},
       science={"phenomena": 4, "bound": 4, "rivals": 4, "replicators": 3}),
    _p("infrastructure_constrained", "Infrastructure-constrained",
       "Small grids that cannot expand, one small compute provider whose datacenter shares a grid with industry, and "
       "a line of low-capacity freight links: energy, compute and transport bind.",
       "Infrastructure-constrained", ("compute_energy", "grid_curtailment", "goods_supply", "goods_wom",
                                      "org_coordination", "education_skills", "freight_energy", "lab_energy",
                                      "energy_cost_pricing", "datacenter_siting"),
       industry={"links": "line"},
       energy={"regime": "scarce", "expand": False},
       compute={"market": "monopoly", "regime": "scarce", "datacenters": "colocated"}),
    _p("resource_abundant", "Resource-abundant",
       "Large expandable grids, many materials and spare compute in every region; nothing physical binds, so "
       "competition is on product attributes and prices.",
       "Resource-abundant", ("science_process", "compute_energy", "goods_supply", "goods_wom", "education_skills",
                             "freight_energy"),
       industry={"materials": 8, "links": "complete"},
       energy={"regime": "abundant", "expand": True},
       compute={"market": "duopoly", "regime": "abundant", "datacenters": "spread"}),
    _p("high_competition", "High-competition",
       "Four assemblers compete for the same components, three compute providers compete on price, and the web has "
       "many shops of the same categories.",
       "High-competition", _C,
       industry={"chain": "wide", "regions": 4, "links": "ring"},
       compute={"market": "competitive", "regime": "normal", "datacenters": "spread"},
       digital={"sites": 28, "shop_share": 0.6}),
    _p("fragmented_social", "Fragmented social networks",
       "People live in contiguous regional blocks, so their contacts stay inside their region; every board is a "
       "regional board; freight is a line. Word of mouth stays local.",
       "Fragmented social networks", ("science_process", "compute_energy", "grid_curtailment", "goods_supply",
                                      "goods_wom", "education_skills", "freight_energy", "datacenter_siting"),
       industry={"links": "line"},
       social={"placement": "blocks", "boards": "regional"}),
    _p("interconnected_communities", "Strongly interconnected communities",
       "People are scattered across regions, so contact links cross every region; boards are global; every region "
       "is linked to every other one.",
       "Strongly interconnected communities", _C,
       industry={"links": "complete"},
       social={"placement": "scattered", "boards": "global"}),
    _p("cooperative_institutions", "Differing institutional structures",
       "Producer firms are cooperatives that share profit by recorded labour hours, component makers share it "
       "equally among members; coordination costs of growing organisations apply.",
       "Differing institutional structures", ("science_process", "compute_energy", "goods_supply", "goods_wom",
                                              "org_coordination", "education_skills", "lab_energy",
                                              "energy_cost_pricing"),
       institutions={"governance": "cooperative", "utilities": "private"}),
    _p("mixed_economy", "Mixed economy",
       "The grid utility and one of the two compute providers are public organisations whose constitutions share "
       "surplus equally among members; every other firm is private.",
       "Mixed economies", ("science_process", "compute_energy", "grid_curtailment", "goods_supply", "goods_wom",
                           "org_coordination", "education_skills", "energy_cost_pricing", "freight_energy"),
       institutions={"governance": "mixed", "utilities": "public"},
       compute={"market": "duopoly", "regime": "normal", "datacenters": "spread"}),
]}

PART_10_1 = ("Digital-first", "Industrial-first", "Research-intensive", "Infrastructure-constrained",
             "Resource-abundant", "High-competition", "Fragmented social networks",
             "Strongly interconnected communities", "Differing institutional structures", "Mixed economies")

# numeric parameters of the categorical regimes (documented here, applied by the generator)
ENERGY_REGIMES = {"scarce": (450, 750), "normal": (6000, 9000), "abundant": (40000, 60000)}
COMPUTE_REGIMES = {"scarce": 24, "normal": 256, "abundant": 4096}         # cores per provider
COMPUTE_MARKETS = {"monopoly": 1, "duopoly": 2, "competitive": 3}


def get(name: str) -> Profile:
    if name not in PROFILES:
        raise KeyError(f"unknown civilisation profile {name!r}; known: {sorted(PROFILES)}")
    return PROFILES[name]


def structural_key(p: Profile) -> dict:
    """The discrete (non-numeric) structure a profile declares."""
    return {"couplings": sorted(p.couplings), "chain": p.industry.get("chain"), "regions": p.industry.get("regions"),
            "links": p.industry.get("links"), "materials": p.industry.get("materials"),
            "energy": p.energy.get("regime"), "grid_expand": bool(p.energy.get("expand")),
            "compute_market": p.compute.get("market"), "compute_regime": p.compute.get("regime"),
            "datacenters": p.compute.get("datacenters"), "placement": p.social.get("placement"),
            "boards": p.social.get("boards"), "governance": p.institutions.get("governance"),
            "utilities": p.institutions.get("utilities"), "phenomena_bound": p.science.get("bound", 0) > 0}


def structural_digest(p: Profile) -> str:
    return digest(structural_key(p))
