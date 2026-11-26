"""Typed event types of the civilisation's own modules (crucible.substrate.bus; MASTER_SPEC Part 2.3).

Civilisation worlds are typed (`params.typed_events`, crucible.civ.generator.recipe): every registered event type of
a module of the world is validated against its schema when it is emitted. This file registers the types emitted by
`civlinks` (host `links`: couplings and the loads, prices and sites they move), `civgoods` (host `goods`) and the
record events of `civgen`. Schemas are closed (`additionalProperties: false`) where the payload is fixed, so a
mechanism that starts writing an undeclared field fails loudly in every civilisation instead of drifting.
"""
from __future__ import annotations

from ..core import schema as S
from ..substrate import bus

_STR = {"type": "string"}
_INT = {"type": "integer"}
_NUM = {"type": "number"}
_BOOL = {"type": "boolean"}
_INT_LIST = {"type": "array", "items": {"type": "integer"}}
_ANY = {"type": "object", "additionalProperties": True}
_PAIR_LIST = {"type": "array", "items": {"type": "array"}}


def _t(name: str, props: dict, required: list[str], owner: str, doc: str, closed: bool = True) -> None:
    sch = S.obj(props, required)
    if not closed:
        sch["additionalProperties"] = True
    bus.event_type(name, sch, owner=owner, doc=doc)


_COUPLING = {"category": _INT, "kind": _STR, "source": _STR, "target": _STR, "observable": _STR, "effect": _STR,
             "mechanism": _STR, "params": _ANY, "pending": _STR, "enabled": _BOOL, "id": _STR}
_t("links.coupling", _COUPLING, ["id", "category", "kind", "source", "target", "enabled"], "civlinks",
   "a coupling record of the civilisation (crucible.civ.couplings.CATALOGUE)")
_t("links.removed", {"id": _STR}, ["id"], "civlinks", "a coupling disabled by an operator (counterfactual)")
_t("links.restored", {"id": _STR}, ["id"], "civlinks", "a removed coupling enabled again")
_t("links.placebo", {"k": _INT}, ["k"], "civlinks", "a no-op event of a placebo arm")
_t("links.dc", {"provider": _STR, "region": _STR, "moved_from": _STR, "curtailed_hours": _INT, "cost": _INT},
   ["provider", "region"], "civlinks", "where a compute provider's datacenter stands (registration or a move)")
_t("links.lab", {"operator": _STR, "region": _STR}, ["operator", "region"], "civlinks",
   "where an apparatus operator stands")
_ENERGY = {"provider": _STR, "region": _STR, "kwh": _NUM, "hour": _INT, "powered_cores": _INT, "loads": _PAIR_LIST,
           "cost": _INT, "reason": _STR}
_t("links.energy", _ENERGY, ["provider", "region", "kwh", "hour"], "civlinks",
   "a datacenter's hourly energy draw from its region's grid (coupling compute_energy)")
_t("links.backup", _ENERGY, ["provider", "region", "kwh", "cost"], "civlinks",
   "a refused datacenter draw covered by backup generation bought outside the world")
_t("links.curtailed", _ENERGY, ["provider", "region", "kwh", "hour"], "civlinks",
   "a refused datacenter draw curtailed the datacenter for the hour (coupling grid_curtailment)")
_t("links.energy_failed", {"provider": _STR, "region": _STR, "error": _STR, "coupling": _STR},
   ["provider", "error"], "civlinks", "an energy draw or a price change that could not be made")
_t("links.load", {"coupling": _STR, "region": _STR, "kwh": _NUM, "payer": _STR, "cost": _INT, "backup": _BOOL,
                  "shipment": _STR, "run": _STR, "refs": _INT_LIST},
   ["coupling", "region", "kwh", "payer", "cost", "backup"], "civlinks",
   "energy drawn for a shipment (freight_energy) or a measurement run (lab_energy)")
_t("links.price_ref", {"provider": _STR, "price": _INT, "energy": _NUM}, ["provider", "price", "energy"], "civlinks",
   "the first priced day of a provider: its price and its energy cost per powered core-hour")
_t("links.repriced", {"provider": _STR, "from": _INT, "to": _INT, "energy_cents_per_core_hour": _NUM, "day": _INT},
   ["provider", "from", "to"], "civlinks", "a provider passed a change of its energy cost on to its compute price")
_t("links.coordination", {"org": _STR, "operators": _INT, "effective_share": _NUM, "hours_lost": _NUM},
   ["org", "operators", "effective_share"], "civlinks", "coordination overhead of an organisation (one day)")
_t("links.rnd", {"org": _STR, "design": _STR, "paper": _STR, "from": {"type": "array"}, "to": {"type": "array"}},
   ["org", "design", "paper", "from", "to"], "civlinks", "a producer trials process settings from a paper")
_t("links.rnd_check", {"org": _STR, "design": _STR, "phenomenon": _STR, "replicated_untried": _INT},
   ["org", "design", "phenomenon", "replicated_untried"], "civlinks",
   "a producer's literature check: independently replicated papers on its machine's physics it has not tried")
_t("links.rnd_result", {"org": _STR, "design": _STR, "paper": _STR, "kept": _BOOL, "base_yield": _NUM,
                        "yield": _NUM}, ["org", "design", "paper", "kept"], "civlinks",
   "the outcome of a producer's R&D trial (kept only if its own yield did not fall)")
_t("goods.wom", {"learner": _INT, "source": _INT, "retail": _STR, "refs": _INT_LIST},
   ["learner", "source", "retail"], "civgoods", "an individual learned of a listing from a contact's purchase")
_t("goods.slot", {"id": _STR, "slot": _INT}, ["id", "slot"], "civgoods", "a retail listing's slot in awareness rows")
