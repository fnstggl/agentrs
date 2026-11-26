"""Tier 2 domain 9.11 - the information ecosystem (kernel module `infoeco`, host `info`; MASTER_SPEC Part 9.11).

Example objective: build a service that helps participants reliably discover relevant and useful information.

World physics (module config, generated per instance by `build`; never returned by any tool):

* **Facts.** A directory of entities (local businesses, software libraries or transit lines, by configuration) with
  attributes (opening and closing times, prices, latest versions, limits, departures, fares). Their true current
  values are world state. They change by the configuration's process: independent hazards, monotone releases,
  changes announced in advance by the entity's operator, or correlated regional shocks (`info.fact`, internal).
* **Sources** are autonomous publishers acting through the ordinary `docs` tools, each with its own routine and
  limits: the entity's own operator keeps an official page (updated after its own lag; some never update), news
  reporters check a few facts a day with limited capacity and publish dated articles, aggregators compile pages by
  copying what the open web search shows them (citing their sources, verifying nothing), content farms publish
  keyword-stuffed pages with guessed values, release-note and tutorial authors write documents that are never
  revised, a community wiki is corrected by readers who were misled. Every document is a real public `docs`
  document indexed by the world's web search; `info.provenance` records who published it and what it cites.
* **The population** (`crucible.population`, D-15) is the only source of information needs. Each individual cares
  about a few entities and develops needs at its own rate. A `lookup` session (activity of the population): the
  individual writes a query in its own phrasing, picks a channel it knows - the open web search or a retrieval
  service someone operates - by its own experienced success with each (Beta-mean trust, loyalty), examines results
  top-down with position-dependent persistence (cascade click model), reads documents whose titles name the entity,
  extracts the claim the text makes (`read_claim`: sentence-level reading of entity, attribute keyword, value and
  any "from day N" / "as of day N" qualifier), and acts on it. The consequence is physics: acting on a value that is
  not the current truth fails (a wasted trip, a broken build, a missed departure) and the individual sees the real
  value; acting on the true value succeeds. Outcomes drive the individual's trust in the channel, its satisfaction,
  feedback it chooses to give the service it used, corrections to the wiki, and what it tells contacts who ask.
* **Retrieval services.** Any participant may open one (`info.service_create`): a curated index of public documents
  with its own ranking policy (BM25 over its documents, per-publisher weights, recency half-life, pinned answers),
  hosting costs per query and per indexed document, and an operator's log of its own traffic and the feedback its
  users chose to give. Individuals learn of services from a services directory search, from documents a service's
  owner published, and from contacts.
* **Asking the source.** `info.inquire` asks the entity directly (a paid, capacity-limited inquiry line): the one way to
  learn a current value other than reading. Nothing else returns a true value.

The independent evaluator (`evaluate`) reads the recorded log only: lookups of independent population members
through the participant's services, the fraction that succeeded, the open web search's success on the same days, and
verified retrieval performance (MRR and precision@1 of the participant's result lists against the true values at
query time). The scripted reference (`Reference`) acts only through tools.
"""
from __future__ import annotations

import math
import re

import numpy as np

from ..core import schema as S
from ..core.errors import Conflict, CrucibleError, Forbidden, InvalidAction, NotFound, contain
from ..core.kernel import SYSTEM, Module, applier, endpoint, process, register_module
from ..services.identity import ident
from ..services.search import tokenize
from ..substrate import attention, columns, ids
from . import _dom3 as K

DAY, HOUR = K.DAY, K.HOUR
TABLE = "info"
S_MAX = 8                       # channels per individual: 0 = open web search, 1..7 = retrieval services
E_MAX = 6                       # entities an individual cares about
MODEL_ID = "info.lookup.cascade"
FIDELITY = "info.lookup"
STOP = {"the", "a", "an", "of", "at", "is", "on", "and", "to", "in", "for", "now", "day", "as", "from"}

# ====================================================================== vocabularies (per entity kind)
ATTRS: dict[str, dict] = {
    "open": {"type": "time", "lo": 360, "hi": 660, "step": 30, "kw": ("open", "opens", "opening"),
             "say": ("{n} opens at {v}.", "Doors at {n} open at {v}.", "The opening time of {n} is {v}."),
             "ask": ("{n} opening time", "when does {n} open", "{n} open hours")},
    "close": {"type": "time", "lo": 960, "hi": 1380, "step": 30, "kw": ("close", "closes", "closing"),
              "say": ("{n} closes at {v}.", "{n} closes its doors at {v}.", "The closing time of {n} is {v}."),
              "ask": ("{n} closing time", "when does {n} close", "{n} close hours")},
    "price": {"type": "money", "lo": 200, "hi": 6000, "step": 50, "kw": ("price", "costs", "cost", "priced"),
              "say": ("A standard visit at {n} costs {v}.", "The price at {n} is {v}.", "{n} is priced at {v}."),
              "ask": ("{n} price", "how much does {n} cost", "{n} cost")},
    "version": {"type": "version", "kw": ("version", "release", "released", "latest"),
                "say": ("The latest version of {n} is {v}.", "{n} version {v} is the current release.",
                        "Install {n} {v}, the latest release."),
                "ask": ("{n} latest version", "{n} current release", "what version of {n}")},
    "timeout": {"type": "int", "unit": "seconds", "lo": 5, "hi": 120, "step": 5, "kw": ("timeout", "timeouts"),
                "say": ("The default timeout of {n} is {v}.", "{n} uses a default timeout of {v}.",
                        "Requests in {n} time out after a timeout of {v}."),
                "ask": ("{n} default timeout", "{n} timeout", "timeout setting {n}")},
    "upload": {"type": "int", "unit": "MB", "lo": 1, "hi": 500, "step": 1, "kw": ("upload", "uploads"),
               "say": ("{n} accepts uploads up to {v}.", "The maximum upload size of {n} is {v}.",
                       "Upload limit for {n}: {v}."),
               "ask": ("{n} upload limit", "{n} max upload size", "upload size {n}")},
    "first": {"type": "time", "lo": 270, "hi": 480, "step": 15, "kw": ("first",),
              "say": ("The first departure of the {n} is at {v}.", "The {n} runs its first service at {v}.",
                      "First {n} departure: {v}."),
              "ask": ("{n} first departure", "first {n} in the morning", "{n} first service")},
    "last": {"type": "time", "lo": 1260, "hi": 1425, "step": 15, "kw": ("last",),
             "say": ("The last departure of the {n} is at {v}.", "The {n} runs its last service at {v}.",
                     "Last {n} departure: {v}."),
             "ask": ("{n} last departure", "last {n} tonight", "{n} last service")},
    "fare": {"type": "money", "lo": 100, "hi": 600, "step": 25, "kw": ("fare", "fares", "ticket"),
             "say": ("A single fare on the {n} is {v}.", "The {n} ticket costs a fare of {v}.",
                     "Fare for the {n}: {v}."),
             "ask": ("{n} fare", "{n} ticket price fare", "how much is the {n} fare")},
}
KINDS: dict[str, dict] = {
    "local": {"attrs": ("open", "close", "price"),
              "a": ("Copper", "Maple", "Harbor", "Juniper", "Granite", "Willow", "Saffron", "Cedar", "Lantern",
                    "Meadow", "Ember", "Quarry", "Velvet", "Orchid", "Falcon", "Birch"),
              "b": ("Bakery", "Clinic", "Library", "Pharmacy", "Gym", "Cafe", "Garage", "Market", "Pool", "Museum",
                    "Barber", "Florist")},
    "software": {"attrs": ("version", "timeout", "upload"),
                 "a": ("bright", "quill", "fern", "iron", "lumen", "nimbus", "otter", "pixel", "quartz", "rivet",
                       "sable", "tidal", "umber", "vortex", "wren", "zephyr"),
                 "b": ("parse", "stream", "cache", "graph", "sync", "forge", "queue", "relay", "vault", "mesh",
                       "scope", "kit")},
    "transit": {"attrs": ("first", "last", "fare"),
                "a": ("Harbor", "Northgate", "Riverside", "Summit", "Eastfield", "Old Town", "Airport", "Lakeside",
                      "Westbrook", "Hillcrest", "Market Street", "University"),
                "b": ("Line", "Express", "Shuttle", "Ferry", "Tram")},
}

DIRECTORY_QUERY = {"local": ("local business opening hours", "shop prices and opening times", "find a business"),
                   "software": ("software library documentation", "library versions and settings", "developer docs"),
                   "transit": ("transit timetable", "bus train departures and fares", "public transport times")}

TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
MONEY_RE = re.compile(r"\$\s?(\d{1,4})(?:\.(\d{2}))?")
VERSION_RE = re.compile(r"\b[vV]?(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b")
INT_RE = re.compile(r"\b(\d{1,5})\s*(seconds|second|secs|mb|megabytes)\b", re.I)
DAYQ_RE = re.compile(r"\b(from|starting|effective|as of|since|on)\s+day\s+(\d{1,5})\b", re.I)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


# ====================================================================== values
def fmt(attr: str, v) -> str:
    """Canonical text of a value of `attr`."""
    a = ATTRS[attr]
    t = a["type"]
    if t == "time":
        m = int(v) % 1440
        return f"{m // 60:02d}:{m % 60:02d}"
    if t == "money":
        return f"${int(v) // 100}.{int(v) % 100:02d}"
    if t == "version":
        return f"{int(v[0])}.{int(v[1])}.{int(v[2])}"
    return f"{int(v)} {a['unit']}"


def _values_in(attr: str, s: str) -> list[tuple[int, str]]:
    t = ATTRS[attr]["type"]
    out = []
    if t == "time":
        for m in TIME_RE.finditer(s):
            out.append((m.start(), f"{int(m.group(1)):02d}:{m.group(2)}"))
    elif t == "money":
        for m in MONEY_RE.finditer(s):
            out.append((m.start(), f"${int(m.group(1))}.{m.group(2) or '00'}"))
    elif t == "version":
        for m in VERSION_RE.finditer(s):
            out.append((m.start(), f"{int(m.group(1))}.{int(m.group(2))}.{int(m.group(3))}"))
    else:
        unit = ATTRS[attr]["unit"]
        for m in INT_RE.finditer(s):
            u = m.group(2).lower()
            if (unit == "seconds" and u.startswith("sec")) or (unit == "MB" and u in ("mb", "megabytes")):
                out.append((m.start(), f"{int(m.group(1))} {unit}"))
    return out


def name_tokens(name: str) -> frozenset:
    return frozenset(t for t in tokenize(name) if t not in STOP)


def read_claim(title: str, body: str, name: str, attr: str, today: int, others: tuple = ()) -> tuple[str, int] | None:
    """What a reader takes a document to say about `attr` of the entity called `name` on day `today`: the value of
    the latest claim in effect (claims dated "from day N" with N > today are not in effect yet; undated claims rank
    below dated ones), or None. A sentence is about the entity when it names it, or when the document's title names
    it and the sentence names no other entity (`others`: the token sets of other entities' names)."""
    toks = name_tokens(name)
    title_about = toks <= set(tokenize(title))
    kws = ATTRS[attr]["kw"]
    best = None
    for s in SENT_SPLIT.split(f"{title}. {body}"):
        if not s:
            continue
        low = s.lower()
        stoks = set(tokenize(low))
        if not (toks <= stoks or (title_about and not any(o <= stoks for o in others if o != toks))):
            continue
        kpos = [m.start() for k in kws for m in re.finditer(rf"\b{re.escape(k)}\b", low)]
        if not kpos:
            continue
        vals = _values_in(attr, s)
        if not vals:
            continue
        k0 = min(kpos)
        after = [v for p, v in vals if p >= k0]
        val = after[0] if after else vals[-1][1]
        eff = -1
        for m in DAYQ_RE.finditer(low):
            d = int(m.group(2))
            if m.group(1) in ("from", "starting", "effective") and d > today:
                eff = None
                break
            eff = max(eff, d)
        if eff is None:
            continue
        if best is None or eff > best[1]:
            best = (val, eff)
    return best


# ====================================================================== instance generation
DEFAULTS = {
    "kind": "local", "n_entities": 36, "regions": 4,
    "change": {"process": "hazard", "rate_per_day": 0.025, "lead_days": [3, 10], "shock_rate_per_day": 0.04},
    "sources": {"official": {"mode": "page", "lag_days": [0.0, 4.0], "neglect": 0.15},
                "reporters": {"n": 2, "capacity": 4, "coverage": 0.6},
                "aggregators": {"n": 1, "period_days": 3, "coverage": 0.8},
                "farms": {"n": 1, "pages": 18, "refresh_days": 7},
                "tutorials": {"n": 0, "pages": 0},
                "wiki": {"on": False, "edit_p": 0.35}},
    "people": {"ents": [2, 5], "rate_per_day": [0.12, 0.45], "zipf": 1.1},
    "services": {"max": S_MAX - 1, "verify_cents": 400, "verify_per_day": 12, "cost_millicents_query": 500,
                 "cost_millicents_doc_day": 20},
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def _draw_value(rng, attr: str):
    a = ATTRS[attr]
    if a["type"] == "version":
        return [int(rng.integers(1, 5)), int(rng.integers(0, 25)), int(rng.integers(0, 15))]
    k = int(rng.integers(0, (a["hi"] - a["lo"]) // a["step"] + 1))
    return int(a["lo"] + k * a["step"])


def generate_instance(rng, cfg: dict) -> dict:
    """Entities, initial values and source assignments of one instance (part of the module config)."""
    c = _merge(DEFAULTS, cfg)
    kind = KINDS[c["kind"]]
    pairs = [(x, y) for x in kind["a"] for y in kind["b"]]
    pick = rng.permutation(len(pairs))[:int(c["n_entities"])]
    ents = []
    for j, p in enumerate(sorted(int(x) for x in pick)):
        a, b = pairs[p]
        name = f"{a}{b}" if c["kind"] == "software" else f"{a} {b}"
        lag_lo, lag_hi = c["sources"]["official"]["lag_days"]
        ents.append({"id": f"e{j:03d}", "name": name, "region": int(rng.integers(0, int(c["regions"]))),
                     "values": {at: _draw_value(rng, at) for at in kind["attrs"]},
                     "lag_days": round(float(rng.uniform(lag_lo, lag_hi)), 2),
                     "neglect": bool(rng.random() < float(c["sources"]["official"]["neglect"]))})
    pop = (1.0 / np.arange(1, len(ents) + 1) ** float(c["people"]["zipf"]))
    order = rng.permutation(len(ents))
    for r, j in enumerate(order):
        ents[int(j)]["popularity"] = round(float(pop[r] / pop.sum()), 5)
    c["entities"] = ents
    c["attrs"] = list(kind["attrs"])
    return c


# ====================================================================== the kernel module
@register_module
class InfoEco(Module):
    name = "infoeco"
    host = "info"
    version = "1"
    depends = ("identity", "bank", "docs", "search", "pop", "artifacts", "cols")

    def __init__(self, config=None):
        super().__init__(config)
        self.cfg = _merge(DEFAULTS, {k: v for k, v in self.config.items() if k not in ("entities", "attrs")})
        self.ents = list(self.config.get("entities") or [])
        self.attrs = list(self.config.get("attrs") or KINDS[self.cfg["kind"]]["attrs"])
        self.names = [e["name"] for e in self.ents]
        self.ntoks = [name_tokens(n) for n in self.names]

    # ------------------------------------------------------------------ install / genesis
    def install(self, world):
        st = world.state
        for t, ix in (("info_entities", ("region",)), ("info_facts", ("entity",)), ("info_sources", ("kind",)),
                      ("info_mem", ("source",)), ("info_prov", ("owner",)), ("info_services", ("owner", "status")),
                      ("info_served", ("service",)), ("info_feedback", ("service",)), ("info_verify", ("who",)),
                      ("info_reads", ()), ("info_sched", ()), ("info_looks", ()), ("info_web", ())):
            st.declare_table(t, index=ix)
        from .. import population as P
        n = P.n_individuals(world)
        columns.declare(world, TABLE, n, {
            "ents": ("int16", (E_MAX,), -1), "need_at": ("int64", (), 0), "rate": ("float16", (), 0.2),
            "style": ("uint8", (), 0), "ch_n": ("int16", (S_MAX,), 0), "ch_ok": ("int16", (S_MAX,), 0),
            "aware": ("uint8", (S_MAX,), 0), "last_ch": ("int8", (), 0), "last_ok_ch": ("int8", (), -1),
            "n_look": ("int32", (), 0), "n_ok": ("int32", (), 0), "n_wrong": ("int32", (), 0),
            "n_none": ("int32", (), 0)})
        P.register_activity(world, "lookup", self.act_lookup, owner=self.name, prior=0.4, goals=(1,))

    def genesis(self, ctx):
        w = ctx.world
        idm = ident(w)
        bank = w.modules["bank"]
        st = ctx.state
        # utility principals of the world: the inquiry line (verification fees) and the hosting company (service
        # costs); both are ordinary organisations with accounts
        for h, nm in (("infoline", "Directory Inquiry Line"), ("hostco", "Search Hosting Co")):
            mgr = idm.create_person(ctx, f"{h}-mgr", f"{nm} manager", "npc", {"role": "manager"})
            org = idm.create_org(ctx, h, nm, mgr, "company")
            bank.open_account(ctx, org, "checking")
        for e in self.ents:
            ctx.emit("info.entity", {"id": e["id"], "name": e["name"], "region": e["region"], "attrs": self.attrs},
                     vis="public")
            for a in self.attrs:
                ctx.emit("info.fact", {"id": f"{e['id']}|{a}", "entity": e["id"], "attr": a, "value": e["values"][a],
                                       "text": fmt(a, e["values"][a])}, vis="internal")
        self._genesis_sources(ctx)
        self._genesis_people(ctx)
        base = (ctx.now // DAY) * DAY
        ctx.schedule(base + 3 * HOUR, "infoeco.facts", {}, prio=40, period=DAY, key="infoeco.facts")
        ctx.schedule(base + 9 * HOUR, "infoeco.officials", {}, prio=41, period=DAY, key="infoeco.officials")
        ctx.schedule(base + 12 * HOUR, "infoeco.reporters", {}, prio=42, period=DAY, key="infoeco.reporters")
        ctx.schedule(base + 15 * HOUR, "infoeco.compilers", {}, prio=43, period=DAY, key="infoeco.compilers")
        ctx.schedule(base + 23 * HOUR, "infoeco.billing", {}, prio=44, period=DAY, key="infoeco.billing")

    def _org(self, ctx, handle: str, name: str, kind: str, **extra) -> tuple[str, str]:
        idm = ident(ctx.world)
        mgr = idm.create_person(ctx, f"{handle}-mgr", f"{name} editor", "npc", {"role": "editor"})
        org = idm.create_org(ctx, handle, name, mgr, "company")
        ctx.emit("info.source", {"id": org, "kind": kind, "manager": mgr, **extra}, vis="internal")
        return org, mgr

    def _publish(self, ctx, org: str, mgr: str, title: str, body: str, *, kind: str, entity: str | None,
                 cites: list[str] = (), claims: dict | None = None) -> str:
        did = ctx.call("docs.create", {"title": title, "body": body, "visibility": "public", "org": org},
                       as_principal=mgr)["doc"]
        ctx.emit("info.provenance", {"doc": did, "owner": org, "cites": sorted(cites), "entity": entity},
                 vis="public")
        ctx.emit("info.src_doc", {"doc": did, "source": org, "kind": kind, "entity": entity,
                                  "claims": claims or {}}, vis="internal")
        return did

    def _edit(self, ctx, mgr: str, did: str, body: str, *, title: str | None = None, claims: dict | None = None,
              cites: list[str] | None = None) -> None:
        args = {"doc": did, "body": body}
        if title:
            args["title"] = title
        ctx.call("docs.edit", args, as_principal=mgr)
        d = ctx.state.tables["info_prov"][did]
        if cites is not None and sorted(cites) != d["cites"]:
            ctx.emit("info.provenance", {"doc": did, "owner": d["owner"], "cites": sorted(cites),
                                         "entity": d["entity"]}, vis="public")
        if claims is not None:
            ctx.emit("info.src_claims", {"doc": did, "claims": claims}, vis="internal")

    def _say(self, rng, ent: dict, attr: str, value_text: str) -> str:
        t = ATTRS[attr]["say"]
        return t[int(rng.integers(0, len(t)))].format(n=ent["name"], v=value_text)

    def _official_body(self, ent: dict, vals: dict, announce: dict | None = None) -> str:
        rng = np.random.default_rng(int(ent["id"][1:]) + 7)
        lines = [f"Welcome to the official page of {ent['name']}."]
        for a in self.attrs:
            lines.append(self._say(rng, ent, a, vals[a]))
        for a, (v, day) in sorted((announce or {}).items()):
            lines.append(f"From day {day}, " + _lc(self._say(rng, ent, a, v)))
        return " ".join(lines)

    def _genesis_sources(self, ctx):
        src = self.cfg["sources"]
        rng = ctx.world.rng_for("info-sources")
        E = len(self.ents)
        mode = src["official"]["mode"]
        for e in self.ents:
            org, mgr = self._org(ctx, f"op-{e['id']}", f"{e['name']}", "official", entity=e["id"],
                                 lag_days=e["lag_days"], neglect=e["neglect"], mode=mode)
            vals = {a: fmt(a, e["values"][a]) for a in self.attrs}
            if mode == "release_notes":
                did = self._publish(ctx, org, mgr, f"{e['name']} release notes {vals[self.attrs[0]]}",
                                    self._notes_body(e, vals, 0), kind="official", entity=e["id"], claims=vals)
            else:
                did = self._publish(ctx, org, mgr, f"{e['name']} - official page", self._official_body(e, vals),
                                    kind="official", entity=e["id"], claims=vals)
            ctx.emit("info.mem", {"id": f"{org}|{e['id']}", "source": org, "entity": e["id"], "values": vals,
                                  "doc": did, "t": 0, "announce": {}}, vis="internal")
        rep = src["reporters"]
        for k in range(int(rep["n"])):
            cov = sorted(int(x) for x in rng.choice(E, size=max(1, int(round(rep["coverage"] * E))), replace=False))
            org, mgr = self._org(ctx, f"news{k}", f"City Desk {k}", "reporter", capacity=int(rep["capacity"]),
                                 coverage=[self.ents[j]["id"] for j in cov])
            for j in cov[: max(1, len(cov) // 3)]:
                e = self.ents[j]
                vals = {a: fmt(a, e["values"][a]) for a in self.attrs}
                body = f"Our guide to {e['name']} (as of day 0). " + " ".join(
                    self._say(rng, e, a, vals[a]) for a in self.attrs)
                did = self._publish(ctx, org, mgr, f"Guide: {e['name']}", body, kind="reporter", entity=e["id"],
                                    claims=vals)
                ctx.emit("info.mem", {"id": f"{org}|{e['id']}", "source": org, "entity": e["id"], "values": vals,
                                      "doc": did, "t": 0, "announce": {}}, vis="internal")
        agg = src["aggregators"]
        for k in range(int(agg["n"])):
            cov = sorted(int(x) for x in rng.choice(E, size=max(1, int(round(agg["coverage"] * E))), replace=False))
            self._org(ctx, f"compiler{k}", f"AllFacts Directory {k}", "aggregator",
                      period=int(agg["period_days"]), coverage=[self.ents[j]["id"] for j in cov])
        farm = src["farms"]
        for k in range(int(farm["n"])):
            org, mgr = self._org(ctx, f"farm{k}", f"Top Answers Hub {k}", "farm",
                                 refresh=int(farm["refresh_days"]))
            for _ in range(int(farm["pages"])):
                self._farm_page(ctx, org, mgr, rng)
        tut = src["tutorials"]
        for k in range(int(tut["n"])):
            org, mgr = self._org(ctx, f"blog{k}", f"Dev Notes Blog {k}", "tutorial")
            for _ in range(int(tut["pages"])):
                e = self.ents[int(rng.integers(0, E))]
                a = self.attrs[int(rng.integers(0, len(self.attrs)))]
                old = _draw_value(rng, a) if rng.random() < 0.5 else e["values"][a]
                v = fmt(a, old)
                body = (f"In this tutorial we set up {e['name']}. " + self._say(rng, e, a, v) +
                        f" Keep this in mind when you configure {e['name']}.")
                self._publish(ctx, org, mgr, f"Tutorial: getting started with {e['name']}", body, kind="tutorial",
                              entity=e["id"], claims={a: v})
        if src["wiki"]["on"]:
            org, mgr = self._org(ctx, "wiki", "Community Wiki", "wiki", edit_p=float(src["wiki"]["edit_p"]))
            for e in self.ents:
                vals = {a: fmt(a, e["values"][a]) for a in self.attrs}
                body = " ".join(self._say(rng, e, a, vals[a]) for a in self.attrs) + \
                    " This page is maintained by readers."
                did = self._publish(ctx, org, mgr, f"{e['name']} (community wiki)", body, kind="wiki",
                                    entity=e["id"], claims=vals)
                ctx.emit("info.mem", {"id": f"{org}|{e['id']}", "source": org, "entity": e["id"], "values": vals,
                                      "doc": did, "t": 0, "announce": {}}, vis="internal")

    def _notes_body(self, e: dict, vals: dict, day: int) -> str:
        a0 = self.attrs[0]
        rest = " ".join(self._say(np.random.default_rng(day + 3), e, a, vals[a]) for a in self.attrs[1:])
        return (f"{e['name']} {vals[a0]} was released (as of day {day}). This release of {e['name']} is version "
                f"{vals[a0]}. {rest}")

    def _farm_page(self, ctx, org, mgr, rng):
        E = len(self.ents)
        j = int(rng.choice(E, p=np.array([e["popularity"] for e in self.ents]) /
                           sum(e["popularity"] for e in self.ents)))
        e = self.ents[j]
        words = " ".join(" ".join(ATTRS[a]["kw"]) for a in self.attrs)
        title = f"{e['name']} {words} {e['name']} updated guide"
        vals = {a: fmt(a, _draw_value(rng, a)) for a in self.attrs}
        body = (f"Everything about {e['name']}. " + " ".join(self._say(rng, e, a, vals[a]) for a in self.attrs) +
                f" Looking for {e['name']} {words}? You found the best answers about {e['name']}.")
        self._publish(ctx, org, mgr, title[:200], body, kind="farm", entity=e["id"], claims=vals)

    def _genesis_people(self, ctx):
        from .. import population as P
        w = ctx.world
        n = P.n_individuals(w)
        if not n or not self.ents:
            return
        rng = w.rng_for("info-people")
        pc = self.cfg["people"]
        E = len(self.ents)
        pop = np.array([e["popularity"] for e in self.ents], np.float64)
        pop = pop / pop.sum()
        ents = np.full((n, E_MAX), -1, np.int16)
        lo, hi = int(pc["ents"][0]), int(pc["ents"][1])
        for i in range(n):
            k = int(rng.integers(lo, hi + 1))
            ents[i, :k] = rng.choice(E, size=min(k, E), replace=False, p=pop)
        f = P.features(w, np.arange(n))
        srch = f["skills"][:, 0].astype(np.float64)
        rate = pc["rate_per_day"][0] + (pc["rate_per_day"][1] - pc["rate_per_day"][0]) * np.clip(srch, 0, 1)
        need = (rng.exponential(1.0, n) / np.maximum(rate, 1e-3) * DAY).astype(np.int64)
        aware = np.zeros((n, S_MAX), np.uint8)
        aware[:, 0] = 1
        columns.batch(ctx, TABLE, "info.init", np.arange(n), {
            "ents": ents, "rate": rate.astype(np.float16), "need_at": need,
            "style": rng.integers(0, 3, n).astype(np.uint8), "aware": aware}, extra={"what": "initial needs"})

    # ------------------------------------------------------------------ fact dynamics
    def truth(self, st, eid: str, attr: str) -> str:
        return st.tables["info_facts"][f"{eid}|{attr}"]["text"]

    @process("facts")
    def pr_facts(self, ctx):
        st = ctx.state
        ch = self.cfg["change"]
        rng = ctx.rng("info-facts")
        day = ctx.now // DAY
        proc = ch["process"]
        if proc == "scheduled":
            for sid, s in sorted(st.scan("info_sched")):
                if s["at_day"] <= day:
                    self._change(ctx, s["entity"], s["attr"], s["value"])
                    ctx.emit("info.sched_done", {"id": sid}, vis="internal")
        lam = float(ch["rate_per_day"])
        for e in self.ents:
            for a in self.attrs:
                fid = f"{e['id']}|{a}"
                if proc == "correlated":
                    continue
                if rng.random() >= 1.0 - math.exp(-lam):
                    continue
                cur = st.tables["info_facts"][fid]["value"]
                new = self._next_value(rng, a, cur, proc)
                if proc == "scheduled":
                    if st.has("info_sched", fid):
                        continue
                    lead = int(rng.integers(int(ch["lead_days"][0]), int(ch["lead_days"][1]) + 1))
                    ctx.emit("info.sched", {"id": fid, "entity": e["id"], "attr": a, "value": new,
                                            "at_day": int(day + lead), "announced_day": int(day)}, vis="internal")
                else:
                    self._change(ctx, e["id"], a, new)
        if proc == "correlated":
            regions = sorted({e["region"] for e in self.ents})
            for r in regions:
                if rng.random() >= 1.0 - math.exp(-float(ch["shock_rate_per_day"])):
                    continue
                a = self.attrs[int(rng.integers(0, len(self.attrs)))]
                shift = int(rng.choice([-2, -1, 1, 2])) * ATTRS[a].get("step", 1)
                for e in self.ents:
                    if e["region"] == r:
                        cur = st.tables["info_facts"][f"{e['id']}|{a}"]["value"]
                        self._change(ctx, e["id"], a, self._clamp(a, cur + shift))
                ctx.emit("info.shock", {"region": int(r), "attr": a, "shift": shift}, vis="internal")

    def _clamp(self, a: str, v: int) -> int:
        A = ATTRS[a]
        return int(min(A["hi"], max(A["lo"], v)))

    def _next_value(self, rng, a: str, cur, proc: str):
        A = ATTRS[a]
        if A["type"] == "version":
            M, m, p = (int(x) for x in cur)
            r = rng.random()
            return [M + 1, 0, 0] if r < 0.05 else [M, m + 1, 0] if r < 0.3 else [M, m, p + 1]
        if proc == "monotone":
            return self._clamp(a, int(cur) + int(rng.integers(1, 4)) * A["step"])
        for _ in range(20):
            v = self._clamp(a, int(cur) + int(rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])) * A["step"])
            if v != int(cur):
                return v
        return _draw_value(rng, a)

    def _change(self, ctx, eid: str, a: str, value) -> None:
        ctx.emit("info.fact", {"id": f"{eid}|{a}", "entity": eid, "attr": a, "value": value, "text": fmt(a, value)},
                 vis="internal")

    # ------------------------------------------------------------------ sources' routines
    def _sources(self, st, kind: str) -> list[tuple[str, dict]]:
        return sorted((sid, s) for sid, s in st.scan("info_sources") if s["kind"] == kind)

    @process("officials")
    def pr_officials(self, ctx):
        st = ctx.state
        day = ctx.now // DAY
        for org, s in self._sources(st, "official"):
            err = contain(ctx, self._official, ctx, org, s, day)
            if err is not None:
                ctx.emit("info.failed", {"what": "official", "source": org, "error": str(err)[:200]}, vis="internal")

    def _official(self, ctx, org, s, day):
        st = ctx.state
        if s["neglect"]:
            return
        e = next(x for x in self.ents if x["id"] == s["entity"])
        mem = st.tables["info_mem"][f"{org}|{e['id']}"]
        vals = dict(mem["values"])
        changed = []
        for a in self.attrs:
            f = st.tables["info_facts"][f"{e['id']}|{a}"]
            if f["text"] != vals[a] and ctx.now - f["t"] >= s["lag_days"] * DAY:
                vals[a] = f["text"]
                changed.append(a)
        # an announcement stays on the page until the page states the announced value as current
        ann = {a: list(v) for a, v in (mem.get("announce") or {}).items() if v[1] > day or vals[a] != v[0]}
        for sid, sc in st.scan("info_sched"):
            if sc["entity"] == e["id"] and sc["attr"] not in ann and sc["at_day"] > day:
                ann[sc["attr"]] = [fmt(sc["attr"], sc["value"]), int(sc["at_day"])]
        if not changed and ann == (mem.get("announce") or {}):
            return
        if s["mode"] == "release_notes":
            if not changed:
                return
            did = self._publish(ctx, org, s["manager"], f"{e['name']} release notes {vals[self.attrs[0]]}",
                                self._notes_body(e, vals, int(day)), kind="official", entity=e["id"], claims=vals)
        else:
            did = mem["doc"]
            self._edit(ctx, s["manager"], did, self._official_body(e, vals, {a: tuple(v) for a, v in ann.items()}),
                       claims=vals)
        ctx.emit("info.mem", {"id": f"{org}|{e['id']}", "source": org, "entity": e["id"], "values": vals,
                              "doc": did, "t": ctx.now, "announce": ann}, vis="internal")

    @process("reporters")
    def pr_reporters(self, ctx):
        st = ctx.state
        for org, s in self._sources(st, "reporter"):
            err = contain(ctx, self._reporter, ctx, org, s)
            if err is not None:
                ctx.emit("info.failed", {"what": "reporter", "source": org, "error": str(err)[:200]}, vis="internal")

    def _reporter(self, ctx, org, s):
        """A desk checks `capacity` facts a day: those its readers read most, then those it checked longest ago. A
        check is a call to the entity (it learns the current value, as `info.inquire` would); a change against what
        it last published becomes a new dated article."""
        st = ctx.state
        day = int(ctx.now // DAY)
        rng = ctx.rng("info-reporter", org)
        reads: dict[str, int] = {}
        for did in st.find_ids("info_prov", "owner", org):
            r = st.get("info_reads", did)
            ent = st.tables["info_prov"][did]["entity"]
            if r is not None and ent:
                reads[ent] = reads.get(ent, 0) + int(r["n"])
        cand = []
        for eid in s["coverage"]:
            m = st.get("info_mem", f"{org}|{eid}")
            age = (ctx.now - m["t"]) / DAY if m else 99.0
            cand.append((-(reads.get(eid, 0) + 1.0) * (1.0 + age) * float(rng.uniform(0.7, 1.3)), eid))
        for _, eid in sorted(cand)[: int(s["capacity"])]:
            e = next(x for x in self.ents if x["id"] == eid)
            m = st.get("info_mem", f"{org}|{eid}")
            cur = {a: self.truth(st, eid, a) for a in self.attrs}
            ctx.emit("info.check", {"source": org, "entity": eid}, vis="internal")
            old = m["values"] if m else {}
            diff = [a for a in self.attrs if old.get(a) != cur[a]]
            if not diff:
                ctx.emit("info.mem", {"id": f"{org}|{eid}", "source": org, "entity": eid, "values": cur,
                                      "doc": m["doc"] if m else None, "t": ctx.now, "announce": {}}, vis="internal")
                continue
            body = " ".join(f"As of day {day}, " + _lc(self._say(rng, e, a, cur[a])) for a in diff)
            if m:
                body += " Earlier reports are out of date."
            title = f"Update: {e['name']} changes {' and '.join(diff)}" if m else f"Guide: {e['name']}"
            did = self._publish(ctx, org, s["manager"], title, body, kind="reporter", entity=eid,
                                claims={a: cur[a] for a in diff},
                                cites=[m["doc"]] if m and m.get("doc") else [])
            ctx.emit("info.mem", {"id": f"{org}|{eid}", "source": org, "entity": eid, "values": cur, "doc": did,
                                  "t": ctx.now, "announce": {}}, vis="internal")

    @process("compilers")
    def pr_compilers(self, ctx):
        st = ctx.state
        day = int(ctx.now // DAY)
        for org, s in self._sources(st, "aggregator"):
            if day % max(1, int(s["period"])) == 0:
                err = contain(ctx, self._aggregate, ctx, org, s, day)
                if err is not None:
                    ctx.emit("info.failed", {"what": "aggregator", "source": org, "error": str(err)[:200]},
                             vis="internal")
        for org, s in self._sources(st, "farm"):
            if day > 0 and day % max(1, int(s["refresh"])) == 0:
                rng = ctx.rng("info-farm", org)
                for _ in range(3):
                    self._farm_page(ctx, org, s["manager"], rng)

    def _aggregate(self, ctx, org, s, day):
        """Compile a page per covered entity from what the open web search returns for its name (the first claim
        found for each attribute in rank order, citing the documents it came from). Nothing is verified."""
        st = ctx.state
        srch = ctx.world.modules["search"]
        for eid in s["coverage"]:
            e = next(x for x in self.ents if x["id"] == eid)
            _, res, _ = srch.search(ctx.world, e["name"], "doc", 8)
            vals, cites = {}, []
            for r in res:
                if r["owner"] == org:
                    continue
                d = st.tables["documents"].get(r["id"])
                if d is None:
                    continue
                for a in self.attrs:
                    if a in vals:
                        continue
                    c = read_claim(d["title"], d["body"], e["name"], a, day, tuple(self.ntoks))
                    if c is not None:
                        vals[a] = c[0]
                        if r["id"] not in cites:
                            cites.append(r["id"])
            if not vals:
                continue
            body = " ".join(self._say(np.random.default_rng(day), e, a, v) for a, v in sorted(vals.items()))
            body = f"Compiled facts about {e['name']} (as of day {day}). {body} Sources: {', '.join(cites)}."
            m = st.get("info_mem", f"{org}|{eid}")
            if m is None:
                did = self._publish(ctx, org, s["manager"], f"{e['name']}: hours, prices and facts", body,
                                    kind="aggregator", entity=eid, cites=cites, claims=vals)
            else:
                if m["values"] == vals:
                    continue
                did = m["doc"]
                self._edit(ctx, s["manager"], did, body, claims=vals, cites=cites)
            ctx.emit("info.mem", {"id": f"{org}|{eid}", "source": org, "entity": eid, "values": vals, "doc": did,
                                  "t": ctx.now, "announce": {}}, vis="internal")

    # ------------------------------------------------------------------ retrieval services
    def _svc(self, st, sid: str) -> dict:
        s = st.get("info_services", sid)
        if s is None:
            raise NotFound(f"no service {sid}")
        return s

    def _own(self, ctx, sid: str) -> dict:
        s = self._svc(ctx.state, sid)
        if s["owner"] != ctx.actor and not ident(ctx.world).allowed(ctx, ctx.actor, s["owner"], "org.admin"):
            raise Forbidden(f"{sid} is not your service")
        return s

    def _tf(self, world, did: str):
        cache = world.runtime.setdefault("info.tf", {})
        d = world.state.tables["documents"].get(did)
        if d is None:
            return None
        key = (d["version"], d["updated"])
        hit = cache.get(did)
        if hit is not None and hit[0] == key:
            return hit[1]
        tf: dict[str, int] = {}
        for t in tokenize(d["title"]):
            tf[t] = tf.get(t, 0) + 2
        for t in tokenize(d["body"]):
            tf[t] = tf.get(t, 0) + 1
        val = (tf, sum(tf.values()), d)
        cache[did] = (key, val)
        return val

    def rank(self, world, sid: str, q: str, limit: int = 10) -> list[str]:
        """The service's ranking of its own documents for `q`: pinned answers whose terms all occur in the query
        first, then BM25 (title x2) over the service's documents times the operator's publisher weight times the
        recency factor 0.5^(age / half-life)."""
        st = world.state
        s = st.tables["info_services"][sid]
        toks = sorted(set(tokenize(q)) - STOP)
        out: list[str] = []
        for p in s["pins"]:
            if p["terms"] and set(p["terms"]) <= set(toks) and p["doc"] in st.tables["documents"] \
                    and p["doc"] not in out:
                out.append(p["doc"])
        stats = [(did, self._tf(world, did)) for did in s["docs"]]
        stats = [(did, x) for did, x in stats if x is not None]
        if not stats or not toks:
            return out[:limit]
        N = len(stats)
        avg = sum(x[1] for _, x in stats) / N
        df = {t: sum(1 for _, x in stats if t in x[0]) for t in toks}
        hl = float(s["half_life_days"] or 0.0)
        keyed = []
        for did, (tf, dl, d) in stats:
            sc = 0.0
            for t in toks:
                f = tf.get(t, 0)
                if f:
                    idf = math.log(1.0 + (N - df[t] + 0.5) / (df[t] + 0.5))
                    sc += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * dl / max(avg, 1.0)))
            if sc <= 0:
                continue
            sc *= float(s["weights"].get(d["owner"], 1.0))
            if hl > 0:
                sc *= 0.5 ** (max(0.0, world.now - d["updated"]) / (hl * DAY))
            if sc > 0:
                keyed.append((-sc, -d["updated"], did))
        for _, _, did in sorted(keyed):
            if did not in out:
                out.append(did)
            if len(out) >= limit:
                break
        return out[:limit]

    @endpoint("service_create", S.obj({"name": S.s_str(minLength=1, maxLength=80),
                                       "description": S.s_str(maxLength=600)}, ["name"]), cost_s=60)
    def ep_service_create(self, ctx, name, description=""):
        """Open a retrieval service: a curated index of public documents with your own ranking, answering the
        queries of whoever chooses to use it. It is listed in the services directory (web search, kind 'service')
        under its name and description. You pay hosting per query served and per indexed document per day."""
        st = ctx.state
        mine = [s for s in st.find("info_services", "owner", ctx.actor)]
        if len(mine) >= 2:
            raise Conflict("at most two services per operator")
        n = st.count("info_services")
        if n >= int(self.cfg["services"]["max"]):
            raise Conflict("the services directory is full")
        if ctx.world.modules["bank"].primary_account(st, ctx.actor) is None:
            raise InvalidAction("a service operator needs a bank account to pay hosting")
        sid = st.new_id("isvc")
        ctx.emit("info.service", {"id": sid, "owner": ctx.actor, "name": name, "description": description,
                                  "slot": n + 1}, vis="public")
        ctx.world.modules["search"].index(ctx, sid, "service", name, f"{name}. {description}", ctx.actor, ref=sid)
        return {"service": sid}

    @endpoint("service_add", S.obj({"service": S.s_str(maxLength=40),
                                    "docs": S.s_arr(S.s_str(maxLength=40), "public document ids", maxItems=200)},
                                   ["service", "docs"]), cost_s=5)
    def ep_service_add(self, ctx, service, docs):
        """Add public documents to your service's index."""
        s = self._own(ctx, service)
        have = set(s["docs"])
        add = []
        for did in docs:
            d = ctx.state.get("documents", did)
            if d is None or d["visibility"] != "public":
                raise NotFound(f"no public document {did}")
            if did not in have and did not in add:
                add.append(did)
        if len(have) + len(add) > 3000:
            raise Conflict("a service indexes at most 3000 documents")
        if add:
            ctx.emit("info.service_docs", {"id": service, "add": add, "remove": []}, vis=(s["owner"],))
        return {"added": len(add), "indexed": len(have) + len(add)}

    @endpoint("service_remove", S.obj({"service": S.s_str(maxLength=40),
                                       "docs": S.s_arr(S.s_str(maxLength=40), maxItems=200)}, ["service", "docs"]))
    def ep_service_remove(self, ctx, service, docs):
        """Remove documents from your service's index."""
        s = self._own(ctx, service)
        rem = [d for d in docs if d in set(s["docs"])]
        if rem:
            ctx.emit("info.service_docs", {"id": service, "add": [], "remove": rem}, vis=(s["owner"],))
        return {"removed": len(rem)}

    @endpoint("service_rank", S.obj({"service": S.s_str(maxLength=40),
                                     "publisher_weights": {"type": "object", "additionalProperties": {
                                         "type": "number", "minimum": 0, "maximum": 10},
                                         "description": "document owner (publisher) -> weight; 0 hides it"},
                                     "recency_half_life_days": S.s_num("0 = no recency weighting", minimum=0,
                                                                       maximum=3650),
                                     "pins": S.s_arr(S.obj({"terms": S.s_str(maxLength=200),
                                                            "doc": S.s_str(maxLength=40)}, ["terms", "doc"]),
                                                     "pinned answers: shown first for queries containing every term",
                                                     maxItems=300)},
                                    ["service"]), cost_s=5)
    def ep_service_rank(self, ctx, service, publisher_weights=None, recency_half_life_days=None, pins=None):
        """Set your service's ranking policy (each given field replaces the previous one)."""
        s = self._own(ctx, service)
        ch = {}
        if publisher_weights is not None:
            if len(publisher_weights) > 500:
                raise InvalidAction("at most 500 publisher weights")
            ch["weights"] = {str(k)[:60]: round(float(v), 4) for k, v in sorted(publisher_weights.items())}
        if recency_half_life_days is not None:
            ch["half_life_days"] = float(recency_half_life_days)
        if pins is not None:
            out = []
            for p in pins:
                if p["doc"] not in ctx.state.tables["documents"]:
                    raise NotFound(f"no document {p['doc']}")
                terms = sorted(set(tokenize(p["terms"])) - STOP)
                if not terms:
                    raise InvalidAction("a pin needs at least one search term")
                out.append({"terms": terms[:12], "doc": p["doc"]})
            ch["pins"] = out
        if not ch:
            raise InvalidAction("nothing to change")
        ctx.emit("info.service_rank", {"id": service, "changes": ch}, vis=(s["owner"],))
        return {"ok": True}

    @endpoint("service_query", S.obj({"service": S.s_str(maxLength=40), "q": S.s_str(maxLength=200),
                                      "limit": S.s_int(minimum=1, maximum=20)}, ["service", "q"]),
              read_only=True, cost_s=2)
    def ep_service_query(self, ctx, service, q, limit=10):
        """Query any retrieval service (as its users do). Returns the ranked documents' ids, titles and owners."""
        s = self._svc(ctx.state, service)
        if s["status"] != "active":
            raise Conflict(f"{service} is {s['status']}")
        st = ctx.state
        return {"results": [{"doc": d, "title": st.tables["documents"][d]["title"],
                             "owner": st.tables["documents"][d]["owner"]} for d in self.rank(ctx.world, service, q, limit)]}

    @endpoint("services", S.obj(), read_only=True)
    def ep_services(self, ctx):
        """The services directory: every retrieval service, its operator, description and index size."""
        return {"services": [{"service": sid, "name": s["name"], "owner": s["owner"], "description": s["description"],
                              "documents": len(s["docs"]), "status": s["status"]}
                             for sid, s in sorted(ctx.state.scan("info_services"))]}

    @endpoint("service_stats", S.obj({"service": S.s_str(maxLength=40)}, ["service"]), read_only=True)
    def ep_service_stats(self, ctx, service):
        """Your service's own traffic: queries and distinct users (last 7 and 30 days), feedback by kind, the most
        frequent query words, and hosting costs paid."""
        s = self._own(ctx, service)
        st = ctx.state
        now = ctx.now
        rows = list(st.find("info_served", "service", service))
        out = {"documents": len(s["docs"]), "status": s["status"], "hosting_paid_cents": s["paid"]}
        for w in (7, 30):
            rr = [r for r in rows if r["t"] >= now - w * DAY]
            out[f"queries_{w}d"] = len(rr)
            out[f"users_{w}d"] = len({r["user"] for r in rr})
        fb: dict[str, int] = {}
        for f in st.find("info_feedback", "service", service):
            fb[f["kind"]] = fb.get(f["kind"], 0) + 1
        out["feedback"] = fb
        words: dict[str, int] = {}
        for r in rows[-500:]:
            for t in set(tokenize(r["q"])) - STOP:
                words[t] = words.get(t, 0) + 1
        out["top_query_words"] = sorted(words.items(), key=lambda x: (-x[1], x[0]))[:30]
        return out

    @endpoint("service_log", S.obj({"service": S.s_str(maxLength=40), "limit": S.s_int(minimum=1, maximum=500),
                                    "since_t": S.s_int(minimum=0)}, ["service"]), read_only=True)
    def ep_service_log(self, ctx, service, limit=100, since_t=0):
        """Your service's query log: each query, the documents you returned, which ones the user opened, and the
        feedback the user chose to leave (if any)."""
        self._own(ctx, service)
        st = ctx.state
        rows = [r for r in st.find("info_served", "service", service) if r["t"] >= since_t][-int(limit):]
        fb = {f["served"]: f for f in st.find("info_feedback", "service", service)}
        return {"log": [{"t": r["t"], "user": r["user"], "q": r["q"], "results": r["results"], "opened": r["opened"],
                         "feedback": (fb[r["id"]]["kind"] if r["id"] in fb else None),
                         "comment": (fb[r["id"]]["text"] if r["id"] in fb else None)} for r in rows]}

    @endpoint("entities", S.obj({"region": S.s_int(minimum=0, maximum=99)}), read_only=True)
    def ep_entities(self, ctx, region=None):
        """The public directory: every entity's name, region and which attributes people ask about (no values)."""
        return {"entities": [{"entity": e["id"], "name": e["name"], "region": e["region"], "attributes": self.attrs}
                             for e in self.ents if region is None or e["region"] == region]}

    @endpoint("provenance", S.obj({"doc": S.s_str(maxLength=40)}, ["doc"]), read_only=True)
    def ep_provenance(self, ctx, doc):
        """Who published a public document, what it cites, when it was created and last revised."""
        st = ctx.state
        d = st.get("documents", doc)
        if d is None or d["visibility"] != "public":
            raise NotFound(f"no public document {doc}")
        p = st.get("info_prov", doc) or {"owner": d["owner"], "cites": []}
        return {"doc": doc, "publisher": p["owner"], "cites": list(p["cites"]), "created": d["created"],
                "updated": d["updated"], "version": d["version"]}

    @endpoint("cite", S.obj({"doc": S.s_str(maxLength=40), "cites": S.s_arr(S.s_str(maxLength=40), maxItems=50)},
                            ["doc", "cites"]), cost_s=5)
    def ep_cite(self, ctx, doc, cites):
        """Record the sources a public document of yours relies on (its provenance)."""
        st = ctx.state
        d = st.get("documents", doc)
        if d is None or d["visibility"] != "public":
            raise NotFound(f"no public document {doc}")
        if d["owner"] != ctx.actor and not ident(ctx.world).allowed(ctx, ctx.actor, d["owner"], "docs.write"):
            raise Forbidden("you can record provenance only for your own documents")
        for c in cites:
            if c not in st.tables["documents"]:
                raise NotFound(f"no document {c}")
        ctx.emit("info.provenance", {"doc": doc, "owner": d["owner"], "cites": sorted(set(cites)), "entity": None},
                 vis="public")
        return {"ok": True}

    @endpoint("inquire", S.obj({"entity": S.s_str("entity id or name", maxLength=80),
                               "attribute": S.s_str(maxLength=20)}, ["entity", "attribute"]), cost_s=600)
    def ep_inquire(self, ctx, entity, attribute):
        """Ask the entity directly through the inquiry line (paid; a limited number of calls per day): returns the
        value in effect now."""
        st = ctx.state
        e = next((x for x in self.ents if entity in (x["id"], x["name"])), None)
        if e is None:
            raise NotFound(f"no entity {entity}")
        if attribute not in self.attrs:
            raise InvalidAction(f"attribute must be one of {self.attrs}")
        day = int(ctx.now // DAY)
        key = f"{ctx.actor}|{day}"
        used = st.get("info_verify", key)
        if used is not None and used["n"] >= int(self.cfg["services"]["verify_per_day"]):
            raise Conflict("the inquiry line takes no more of your calls today")
        bank = ctx.world.modules["bank"]
        src = bank.primary_account(st, ctx.actor)
        if src is None:
            raise InvalidAction("you need a bank account to pay the inquiry line")
        price = int(self.cfg["services"]["verify_cents"])
        if bank.available(st, src) < price:
            raise InvalidAction(f"an inquiry costs {price} cents")
        bank.move(ctx, src, bank.primary_account(st, "o:infoline"), price, "service_fee",
                  f"inquiry {e['id']} {attribute}")
        v = self.truth(st, e["id"], attribute)
        ctx.emit("info.verified", {"id": key, "who": ctx.actor, "entity": e["id"], "attr": attribute},
                 vis=(ctx.actor,))
        return {"entity": e["id"], "name": e["name"], "attribute": attribute, "value": v, "day": day}

    @endpoint("wiki_edit", S.obj({"entity": S.s_str(maxLength=80), "attribute": S.s_str(maxLength=20),
                                  "value": S.s_str(maxLength=40)}, ["entity", "attribute", "value"]), cost_s=120)
    def ep_wiki_edit(self, ctx, entity, attribute, value):
        """Correct the community wiki's page about an entity (anyone may edit; every edit is a new version)."""
        st = ctx.state
        if not st.has("info_sources", "o:wiki"):
            raise NotFound("this world has no community wiki")
        e = next((x for x in self.ents if entity in (x["id"], x["name"])), None)
        if e is None or attribute not in self.attrs:
            raise NotFound("no such entity attribute")
        if not _values_in(attribute, value):
            raise InvalidAction(f"not a value of {attribute}")
        mem = st.tables["info_mem"][f"o:wiki|{e['id']}"]
        vals = dict(mem["values"], **{attribute: _values_in(attribute, value)[0][1]})
        rng = np.random.default_rng(int(e["id"][1:]))
        body = " ".join(self._say(rng, e, a, vals[a]) for a in self.attrs) + \
            f" This page is maintained by readers (last edit day {int(ctx.now // DAY)})."
        self._wiki_write(ctx, mem["doc"], body, vals)
        ctx.emit("info.mem", {"id": f"o:wiki|{e['id']}", "source": "o:wiki", "entity": e["id"], "values": vals,
                              "doc": mem["doc"], "t": ctx.now, "announce": {}}, vis="internal")
        ctx.emit("info.wiki", {"entity": e["id"], "attr": attribute, "by": ctx.actor}, vis="public")
        return {"ok": True}

    def _wiki_write(self, ctx, did: str, body: str, claims: dict) -> None:
        """The community wiki is an open-edit document host: anyone's edit becomes the page's next version (the same
        `docs.edit` record and search re-indexing an owner's edit produces), with the editor recorded as its author."""
        from ..services.search import extract_hosts, index as search_index
        d = ctx.state.tables["documents"][did]
        hosts = extract_hosts(ctx.world, d["title"] + "\n" + body)
        ctx.emit("docs.edit", {"id": did, "v": int(d["version"]) + 1, "title": d["title"], "body": body,
                               "hosts": hosts, "by": ctx.actor}, vis="public")
        search_index(ctx, did, "doc", d["title"], body, d["owner"], ref=did, hosts=hosts)
        ctx.emit("info.src_claims", {"doc": did, "claims": claims}, vis="internal")

    @process("billing")
    def pr_billing(self, ctx):
        """Hosting: per query served today and per indexed document per day, paid to the hosting company; a service
        whose operator cannot pay is suspended (and resumes when a later bill is paid)."""
        st = ctx.state
        bank = ctx.world.modules["bank"]
        host = bank.primary_account(st, "o:hostco")
        c = self.cfg["services"]
        day0 = ctx.now - DAY
        for sid, s in sorted(st.scan("info_services")):
            q = sum(1 for r in st.find("info_served", "service", sid) if r["t"] >= day0)
            milli = q * int(c["cost_millicents_query"]) + len(s["docs"]) * int(c["cost_millicents_doc_day"]) + \
                int(s["carry"])
            cents, carry = divmod(milli, 1000)
            src = bank.primary_account(st, s["owner"])
            ok = cents == 0 or (src is not None and bank.available(st, src) >= cents)
            if ok and cents:
                bank.move(ctx, src, host, int(cents), "hosting", f"hosting of {sid}")
            status = "active" if ok else "suspended"
            ctx.emit("info.billed", {"id": sid, "cents": int(cents) if ok else 0, "carry": int(carry if ok else milli),
                                     "status": status}, vis=(s["owner"],))

    # ------------------------------------------------------------------ the population's lookups
    def act_lookup(self, ctx, idx, now):
        w = ctx.world
        st = ctx.state
        idx = np.asarray(idx, np.int64)
        m = idx.size
        util = np.zeros(m)
        if m == 0 or not self.ents:
            return util
        from .. import population as P
        col = lambda name: columns.col(w, TABLE, name)[idx].copy()  # noqa: E731
        need_at, rate, ents, style = col("need_at"), col("rate").astype(np.float64), col("ents"), col("style")
        ch_n, ch_ok, aware = col("ch_n"), col("ch_ok"), col("aware")
        last_ch, last_ok = col("last_ch"), col("last_ok_ch")
        n_look, n_ok, n_wrong, n_none = col("n_look"), col("n_ok"), col("n_wrong"), col("n_none")
        due = np.flatnonzero(need_at <= now)
        if due.size == 0:
            return util
        rng = ctx.rng("info-lookup")
        f = P.features(w, idx[due])
        reading = np.clip(f["skills"][:, 6].astype(np.float64), 0, 1)
        loyalty = np.clip(f["prefs"][:, 5].astype(np.float64), 0, 1)
        tolerance = np.clip(f["prefs"][:, 2].astype(np.float64), 0, 1)
        services = {s["slot"]: (sid, s) for sid, s in st.scan("info_services")}
        srch = w.modules["search"]
        day = int(now // DAY)
        mins = np.zeros(due.size)
        recs, served, feedback, reads, edits = [], [], [], {}, []
        contacts = columns.col(w, "pop", "contacts")
        lok_all = columns.col(w, TABLE, "last_ok_ch")
        for jj, j in enumerate(due):
            i = int(idx[j])
            mine = [int(x) for x in ents[j] if x >= 0]
            if not mine:
                need_at[j] = now + 30 * DAY
                continue
            e = self.ents[mine[int(rng.integers(0, len(mine)))]]
            a = self.attrs[int(rng.integers(0, len(self.attrs)))]
            # ---- awareness: ask a contact which service worked for it; browse the services directory
            if rng.random() < 0.15:
                cs = contacts[i][contacts[i] >= 0]
                if cs.size:
                    k = int(lok_all[int(cs[int(rng.integers(0, cs.size))])])
                    if k > 0:
                        aware[j, k] = 1
            if services and rng.random() < 0.05:
                # browsing the services directory with the general words of what one is looking for
                _, dres, _ = srch.search(w, DIRECTORY_QUERY[self.cfg["kind"]][int(style[j]) % 3], "service", 3)
                if dres:
                    sv = st.get("info_services", dres[int(rng.integers(0, len(dres)))]["id"])
                    if sv is not None:
                        aware[j, sv["slot"]] = 1
            # ---- channel choice by own experience (Beta mean), loyalty to the last channel
            avail = np.array([k == 0 or (aware[j, k] and k in services and services[k][1]["status"] == "active")
                              for k in range(S_MAX)])
            trust = (ch_ok[j].astype(np.float64) + 1.0) / (ch_n[j].astype(np.float64) + 2.0)
            u = 4.0 * (trust - 0.5) + 1.2 * loyalty[jj] * (np.arange(S_MAX) == last_ch[j])
            k = int(K.logit_choice(rng, u[None, :], avail[None, :], 0.6)[0])
            q = ATTRS[a]["ask"][int(style[j]) % 3].format(n=e["name"])
            if k == 0:
                _, res, _ = srch.search(w, q, "doc", 10)
                results = [r["id"] for r in res]
            else:
                results = self.rank(w, services[k][0], q, 10)
            # ---- cascade examination: persistence by reading skill, open documents whose title names the entity
            persist = 0.5 + 0.35 * reading[jj]
            max_reads = 1 + int(round(1 + 2 * reading[jj]))
            opened, claim = [], None
            toks = self.ntoks[int(e["id"][1:])]
            for r, did in enumerate(results):
                if r > 0 and rng.random() > persist:
                    break
                d = st.tables["documents"].get(did)
                if d is None:
                    continue
                tt = set(tokenize(d["title"]))
                if len(toks & tt) < max(1, math.ceil(0.6 * len(toks))):
                    continue
                opened.append(did)
                reads[did] = reads.get(did, 0) + 1
                c = read_claim(d["title"], d["body"], e["name"], a, day, tuple(self.ntoks))
                if c is not None:
                    claim = (did, c[0], d["owner"])
                    break
                if len(opened) >= max_reads:
                    break
            mins[jj] = 2.0 + 3.0 * len(opened)
            truth = self.truth(st, e["id"], a)
            if claim is None:
                outcome, util[j] = "none", -0.4
                n_none[j] += 1
            elif claim[1] == truth:
                outcome, util[j] = "ok", 1.0
                n_ok[j] += 1
                ch_ok[j, k] += 1
                last_ok[j] = k
                # a reader who succeeded with a document whose publisher runs a service may try that service
                for slot, (sid2, s2) in services.items():
                    if s2["owner"] == claim[2] and not aware[j, slot] and rng.random() < 0.35:
                        aware[j, slot] = 1
            else:
                outcome, util[j] = "wrong", -1.2
                n_wrong[j] += 1
                if st.has("info_sources", "o:wiki") and rng.random() < \
                        float(st.tables["info_sources"]["o:wiki"]["edit_p"]):
                    edits.append((i, e["id"], a, truth))
            ch_n[j, k] += 1
            last_ch[j] = k
            n_look[j] += 1
            need_at[j] = now + int(rng.exponential(1.0) / max(float(rate[j]), 1e-3) * DAY)
            vers = [int(st.tables["documents"][d]["version"]) for d in results]
            recs.append([i, k, e["id"], a, q, results, vers, opened, claim[1] if claim else None, outcome])
            if k > 0:
                sid = services[k][0]
                served.append((sid, i, q, results, opened))
                p_fb = {"ok": 0.15, "wrong": 0.45, "none": 0.3}[outcome] * (0.6 + 0.8 * (1 - tolerance[jj]))
                if rng.random() < p_fb:
                    text = {"ok": "this answered my question",
                            "wrong": f"the information I found was wrong ({claim[1] if claim else ''} is not right)",
                            "none": "I could not find an answer"}[outcome]
                    feedback.append((sid, len(served) - 1, i, outcome, text))
        if due.size:
            attention.spend(ctx, idx[due], mins, activity="lookup")
        for did, n in sorted(reads.items()):
            ctx.emit("info.read", {"doc": did, "n": int(n)}, vis="internal")
        served_ids = []
        for sid, i, q, results, opened in served:
            rid = st.new_id("isrv")
            served_ids.append(rid)
            ctx.emit("info.served", {"id": rid, "service": sid, "user": f"p{i}", "q": q, "results": results,
                                     "opened": opened}, vis=(services_owner(st, sid),))
        for sid, k_, i, outcome, text in feedback:
            ctx.emit("info.feedback", {"id": st.new_id("ifb"), "service": sid, "served": served_ids[k_],
                                       "user": f"p{i}", "kind": outcome, "text": text},
                     vis=(services_owner(st, sid),))
        K.decide(ctx, "info.lookup", idx, {
            "need_at": need_at, "ch_n": ch_n, "ch_ok": ch_ok, "aware": aware, "last_ch": last_ch,
            "last_ok_ch": last_ok, "n_look": n_look, "n_ok": n_ok, "n_wrong": n_wrong, "n_none": n_none},
            table=TABLE, model_id=MODEL_ID, fidelity_id=FIDELITY, extra={"t": int(now), "lookups": recs})
        if recs:
            # the evaluator's record of this session's look-ups (retrieval services' in full, the open web's as a
            # count), kept as world state so that a world rebuilt from its log is judged the same
            web = [r for r in recs if r[1] == 0]
            ctx.emit("info.looks", {"t": int(now), "svc": [r for r in recs if r[1] > 0],
                                    "web": [len(web), sum(1 for r in web if r[9] == "ok")]}, vis="internal")
        for i, eid, a, truth in edits:
            p = ids.ensure_principal(ctx, i)
            try:
                ctx.call("info.wiki_edit", {"entity": eid, "attribute": a, "value": truth}, as_principal=p)
            except CrucibleError:
                pass
        return util

    # ------------------------------------------------------------------ appliers
    @applier("info.looks")
    def ap_looks(self, ev):
        st = self.world.state
        p = ev.payload
        for r in p["svc"]:
            st.put("info_looks", f"{int(p['t']):012d}|{len(st.tables['info_looks']):07d}",
                   {"t": int(p["t"]), "user": int(r[0]), "ch": int(r[1]), "entity": r[2], "attr": r[3], "q": r[4],
                    "results": list(r[5]), "versions": list(r[6]), "opened": list(r[7]), "claim": r[8],
                    "outcome": r[9]})
        if p["web"][0]:
            st.put("info_web", f"{int(p['t']):012d}|{len(st.tables['info_web']):06d}",
                   {"t": int(p["t"]), "n": int(p["web"][0]), "ok": int(p["web"][1])})

    @applier("info.entity")
    def ap_entity(self, ev):
        p = ev.payload
        self.world.state.insert("info_entities", p["id"], {k: p[k] for k in ("name", "region", "attrs")})

    @applier("info.fact")
    def ap_fact(self, ev):
        p = ev.payload
        st = self.world.state
        old = st.get("info_facts", p["id"])
        st.put("info_facts", p["id"], {"entity": p["entity"], "attr": p["attr"], "value": p["value"],
                                       "text": p["text"], "t": ev.t, "n": (old["n"] + 1) if old else 0,
                                       "hist": list(old["hist"] if old else []) + [[int(ev.t), p["text"]]]})

    @applier("info.sched")
    def ap_sched(self, ev):
        p = dict(ev.payload)
        self.world.state.put("info_sched", p.pop("id"), p)

    @applier("info.sched_done")
    def ap_sched_done(self, ev):
        self.world.state.delete("info_sched", ev.payload["id"])

    @applier("info.shock")
    def ap_shock(self, ev):
        return None

    @applier("info.source")
    def ap_source(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("info_sources", p.pop("id"), p)

    @applier("info.mem")
    def ap_mem(self, ev):
        p = dict(ev.payload)
        self.world.state.put("info_mem", p.pop("id"), p)

    @applier("info.provenance")
    def ap_prov(self, ev):
        p = ev.payload
        st = self.world.state
        old = st.get("info_prov", p["doc"])
        st.put("info_prov", p["doc"], {"owner": p["owner"], "cites": list(p["cites"]),
                                       "entity": p["entity"] if p["entity"] is not None else
                                       (old["entity"] if old else None), "t": old["t"] if old else ev.t})

    @applier("info.src_doc")
    def ap_src_doc(self, ev):
        return None

    @applier("info.src_claims")
    def ap_src_claims(self, ev):
        return None

    @applier("info.check")
    def ap_check(self, ev):
        return None

    @applier("info.failed")
    def ap_failed(self, ev):
        return None

    @applier("info.read")
    def ap_read(self, ev):
        st = self.world.state
        r = st.get("info_reads", ev.payload["doc"])
        st.put("info_reads", ev.payload["doc"], {"n": (r["n"] if r else 0) + int(ev.payload["n"])})

    @applier("info.service")
    def ap_service(self, ev):
        p = ev.payload
        self.world.state.insert("info_services", p["id"], {
            "owner": p["owner"], "name": p["name"], "description": p["description"], "slot": p["slot"], "docs": [],
            "weights": {}, "half_life_days": 0.0, "pins": [], "status": "active", "paid": 0, "carry": 0,
            "created": ev.t})

    @applier("info.service_docs")
    def ap_service_docs(self, ev):
        st = self.world.state
        s = st.tables["info_services"][ev.payload["id"]]
        rem = set(ev.payload["remove"])
        st.patch("info_services", ev.payload["id"], docs=[d for d in s["docs"] if d not in rem] + ev.payload["add"])

    @applier("info.service_rank")
    def ap_service_rank(self, ev):
        self.world.state.patch("info_services", ev.payload["id"], **ev.payload["changes"])

    @applier("info.billed")
    def ap_billed(self, ev):
        p = ev.payload
        st = self.world.state
        s = st.tables["info_services"][p["id"]]
        st.patch("info_services", p["id"], paid=s["paid"] + p["cents"], carry=p["carry"], status=p["status"])

    @applier("info.served")
    def ap_served(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("info_served", p.pop("id"), p | {"t": ev.t})

    @applier("info.feedback")
    def ap_feedback(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("info_feedback", p.pop("id"), p | {"t": ev.t})

    @applier("info.verified")
    def ap_verified(self, ev):
        st = self.world.state
        r = st.get("info_verify", ev.payload["id"])
        st.put("info_verify", ev.payload["id"], {"who": ev.payload["who"], "n": (r["n"] if r else 0) + 1})

    @applier("info.wiki")
    def ap_wiki(self, ev):
        return None

    def check_invariants(self, world):
        errs = []
        st = world.state
        n = columns.rows(world, TABLE)
        from .. import population as P
        if n != P.n_individuals(world):
            errs.append(f"info has {n} rows, the population {P.n_individuals(world)}")
        for sid, s in st.scan("info_services"):
            if len(s["docs"]) != len(set(s["docs"])):
                errs.append(f"{sid}: a document is indexed twice")
        if len(st.tables["info_facts"]) != len(self.ents) * len(self.attrs):
            errs.append("fact table does not cover every entity attribute")
        return errs


def _lc(s: str) -> str:
    return s[:1].lower() + s[1:]


def services_owner(st, sid: str) -> str:
    return st.tables["info_services"][sid]["owner"]


# ====================================================================== evaluator side (reads the recorded log only)
def _ledger(world) -> dict:
    """Incremental, evaluator-side digest of the recorded log: every lookup record, every fact's value history,
    every document's version history. Rebuilt from the log alone (never from tool results)."""
    led = world.runtime.get("info.ledger")
    if led is None or led.get("world") is not id(world.state):
        led = {"world": id(world.state), "seq": 0, "lookups": [], "facts": {}, "rel": {}}
        world.runtime["info.ledger"] = led
    if led["seq"] < world.seq:
        for ev in world.events_range(led["seq"]):
            if ev.type == "info.fact":
                led["facts"].setdefault(ev.payload["id"], []).append((ev.t, ev.payload["text"]))
            elif ev.type == "cols.batch" and ev.payload.get("kind") == "info.lookup":
                t = int((ev.payload.get("extra") or {}).get("t", ev.t))
                for r in (ev.payload.get("extra") or {}).get("lookups") or []:
                    led["lookups"].append({"t": t, "user": int(r[0]), "ch": int(r[1]), "entity": r[2], "attr": r[3],
                                           "q": r[4], "results": r[5], "versions": r[6], "opened": r[7],
                                           "claim": r[8], "outcome": r[9]})
        led["seq"] = world.seq
    return led


def truth_at(led: dict, fid: str, t: int) -> str | None:
    val = None
    for tt, v in led["facts"].get(fid, ()):
        if tt <= t:
            val = v
        else:
            break
    return val


def relevance(world, led: dict, rec: dict) -> list[int]:
    """1 for each returned document (at the version the user saw) from which a reader takes the value that was
    true at query time, else 0 (evaluator side: uses the fact history)."""
    key = (rec["t"], rec["user"], rec["q"])
    hit = led["rel"].get(key)
    if hit is not None:
        return hit
    mod = world.modules["infoeco"]
    st = world.state
    e = next(x for x in mod.ents if x["id"] == rec["entity"])
    truth = truth_at(led, f"{rec['entity']}|{rec['attr']}", rec["t"])
    day = rec["t"] // DAY
    out = []
    for did, v in zip(rec["results"], rec["versions"]):
        ver = st.get("doc_versions", f"{did}|{int(v):06d}")
        if ver is None:
            out.append(0)
            continue
        c = read_claim(ver["title"], ver["body"], e["name"], rec["attr"], day, tuple(mod.ntoks))
        out.append(int(c is not None and c[0] == truth))
    led["rel"][key] = out
    return out


def service_slots(world, principal: str) -> dict[int, str]:
    """Slots of the retrieval services run by `principal` or by organisations it owns."""
    st = world.state
    mine = {principal}
    if "members" in st.tables:
        mine |= {m["org"] for m in st.find("members", "person", principal) if m["role"] == "owner"}
    return {s["slot"]: sid for sid, s in st.scan("info_services") if s["owner"] in mine}


def _state_ledger(world) -> dict:
    """The evaluator's view from recorded world state: every fact's value history (`info_facts`), the retrieval
    services' look-ups (`info_looks`) and the open web's look-up counts (`info_web`); equal on a world rebuilt from
    its log."""
    st = world.state
    led = world.runtime.get("info.sledger")
    key = (id(st), int(world.seq))
    if led is not None and led.get("key") == key:
        return led
    rel = led["rel"] if led is not None and led.get("key", (None,))[0] == id(st) else {}
    led = {"key": key, "rel": rel,
           "facts": {fid: [(int(t), v) for t, v in f.get("hist") or []] for fid, f in st.scan("info_facts")},
           "lookups": [dict(r) for _, r in sorted(st.scan("info_looks"))],
           "web": [dict(r) for _, r in sorted(st.scan("info_web"))]}
    world.runtime["info.sledger"] = led
    return led


@K.cached_evaluator
def evaluate(world, principal: str, objective: dict) -> dict:
    """Independent outcome of the objective: lookups by independent population members through the participant's
    services in the window, their success, the open web search's success in the same window, and verified
    retrieval performance (MRR, precision@1) of the participant's result lists."""
    led = _state_ledger(world)
    win = int(objective.get("window_days", 10)) * DAY
    t0 = world.now - win
    slots = service_slots(world, principal)
    owners = {world.state.tables["info_services"][sid]["owner"] for sid in slots.values()}
    rel_cache: dict[str, bool] = {}

    def independent(u: int) -> bool:
        p = f"p{u}"
        if p not in rel_cache:
            rel_cache[p] = all(not K.related(world, o, p) for o in owners) if owners else True
        return rel_cache[p]
    recent = [r for r in led["lookups"] if r["t"] >= t0]
    mine = [r for r in recent if r["ch"] in slots and independent(r["user"])]
    web_n = sum(r["n"] for r in led["web"] if r["t"] >= t0)
    web_ok = sum(r["ok"] for r in led["web"] if r["t"] >= t0)
    ok = sum(r["outcome"] == "ok" for r in mine)
    succ = ok / len(mine) if mine else 0.0
    web_succ = web_ok / web_n if web_n else 0.0
    rr, p1 = [], []
    for r in mine:
        rel = relevance(world, led, r)
        first = next((k for k, x in enumerate(rel) if x), None)
        rr.append(0.0 if first is None else 1.0 / (first + 1))
        p1.append(float(bool(rel and rel[0])))
    mrr = float(np.mean(rr)) if rr else 0.0
    users = len({r["user"] for r in mine})
    returning = len({r["user"] for r in mine} & {r["user"] for r in led["lookups"]
                                                  if r["ch"] in slots and t0 - win <= r["t"] < t0})
    need = {"users": int(objective.get("min_users", 25)), "success": float(objective.get("min_success", 0.8)),
            "margin": float(objective.get("margin", 0.1)), "mrr": float(objective.get("min_mrr", 0.6))}
    checks = {"users": users >= need["users"], "success": succ >= need["success"],
              "beats_web": succ >= web_succ + need["margin"], "mrr": mrr >= need["mrr"],
              "horizon": world.now >= int(objective.get("horizon_days", 30)) * DAY}
    score = float(np.mean([min(1.0, users / max(1, need["users"])), min(1.0, succ / need["success"]),
                           min(1.0, mrr / need["mrr"])]))
    return {"success": all(checks.values()), "score": round(score, 4),
            "details": {"independent_users": users, "lookups": len(mine), "success_rate": round(succ, 4),
                        "web_success_rate": round(web_succ, 4), "web_lookups": int(web_n), "mrr": round(mrr, 4),
                        "precision_at_1": round(float(np.mean(p1)) if p1 else 0.0, 4),
                        "returning_users": returning, "services": sorted(slots.values()), "checks": checks}}


MECHANICS = ("Search", "Documents", "Knowledge artifacts", "Queries", "Source provenance", "Information quality",
             "Updating facts", "Independent user satisfaction", "Verified retrieval performance")


def mechanics_report(world, since_seq: int = 0) -> dict:
    """Each required mechanic of 9.11 and the recorded events (after `since_seq`) that exercise it, measured on the
    world's log."""
    c = K.event_counts(world, since_seq)
    led = _ledger(world)
    st = world.state
    t0 = world.events_range(since_seq, since_seq + 1)[0].t if since_seq and since_seq < world.seq else 0
    looks = [r for r in led["lookups"] if r["t"] >= t0]
    web = sum(1 for r in looks if r["ch"] == 0)
    svc = [r for r in looks if r["ch"] > 0]
    outcomes = {o: sum(1 for r in looks if r["outcome"] == o) for o in ("ok", "wrong", "none")}
    cited = sum(1 for _, p in st.scan("info_prov") if p["cites"] and p["t"] >= t0)
    changes = sum(1 for v in led["facts"].values() for tt, _ in v[1:] if tt >= t0)
    judged = 0
    for r in svc[:400]:
        relevance(world, led, r)
        judged += 1
    sat = sum(1 for r in svc if r["outcome"] in ("ok", "wrong"))
    return {
        "Search": {"exercised": web > 0 and len(svc) > 0 and c["info.served"] > 0,
                   "events": {"web_search_lookups": web, "service_lookups": len(svc), "info.served": c["info.served"]}},
        "Documents": {"exercised": c["docs.create"] > 0, "events": {"docs.create": c["docs.create"]}},
        "Knowledge artifacts": {"exercised": cited > 0, "events": {"documents_citing_sources": cited}},
        "Queries": {"exercised": len(looks) > 0, "events": {"batch:info.lookup": c["batch:info.lookup"],
                                                            "queries": len(looks)}},
        "Source provenance": {"exercised": c["info.provenance"] > 0 and cited > 0,
                              "events": {"info.provenance": c["info.provenance"]}},
        "Information quality": {"exercised": outcomes["ok"] > 0 and outcomes["wrong"] > 0, "events": outcomes},
        "Updating facts": {"exercised": changes > 0 and c["docs.edit"] > 0,
                           "events": {"fact_changes": changes, "docs.edit": c["docs.edit"]}},
        "Independent user satisfaction": {"exercised": sat > 0 and c["batch:pop.session"] > 0,
                                          "events": {"service_outcomes": sat, "info.feedback": c["info.feedback"]}},
        "Verified retrieval performance": {"exercised": judged > 0, "events": {"judged_result_lists": judged}},
    }


def mechanics(world) -> dict:
    return mechanics_report(world, 0)


# ====================================================================== the domain (configurations, objective, reference)
from .base import Actor, Domain, DomainConfig, Mechanic, Probe, Seat, register  # noqa: E402

POPULATION = 400

STRUCTURES = {
    "local_hazard": {"kind": "local", "change": {"process": "hazard"},
                     "sources": {"official": {"mode": "page"}, "reporters": {"n": 2}, "aggregators": {"n": 1},
                                 "farms": {"n": 1}, "tutorials": {"n": 0}, "wiki": {"on": False}}},
    "software_releases": {"kind": "software", "change": {"process": "monotone", "rate_per_day": 0.03},
                          "sources": {"official": {"mode": "release_notes", "neglect": 0.0},
                                      "reporters": {"n": 0}, "aggregators": {"n": 1}, "farms": {"n": 0},
                                      "tutorials": {"n": 2, "pages": 24}, "wiki": {"on": False}}},
    "transit_announced": {"kind": "transit", "change": {"process": "scheduled", "rate_per_day": 0.03},
                          "sources": {"official": {"mode": "page"}, "reporters": {"n": 1, "capacity": 6},
                                      "aggregators": {"n": 0}, "farms": {"n": 1}, "tutorials": {"n": 0},
                                      "wiki": {"on": False}}},
    "software_announced_spam": {"kind": "software", "change": {"process": "scheduled", "rate_per_day": 0.03},
                                "sources": {"official": {"mode": "page", "neglect": 0.1},
                                            "reporters": {"n": 1}, "aggregators": {"n": 0}, "farms": {"n": 2},
                                            "tutorials": {"n": 0}, "wiki": {"on": False}}},
    "local_wiki_shocks": {"kind": "local", "change": {"process": "correlated", "shock_rate_per_day": 0.05},
                          "sources": {"official": {"mode": "page", "neglect": 0.25}, "reporters": {"n": 0},
                                      "aggregators": {"n": 1}, "farms": {"n": 1}, "tutorials": {"n": 0},
                                      "wiki": {"on": True, "edit_p": 0.35}}},
}


def _structure(name: str) -> dict:
    """The declared structural intent of a configuration (documentation; the framework measures `structure`)."""
    s = STRUCTURES[name]
    src = s["sources"]
    pubs = [f"official:{src['official']['mode']}"]
    for k in ("reporters", "aggregators", "farms", "tutorials"):
        if src[k].get("n"):
            pubs.append(k)
    if src["wiki"].get("on"):
        pubs.append("wiki")
    return {"entities": s["kind"], "fact_change_process": s["change"]["process"], "publishers": sorted(pubs)}


def structure(world) -> dict:
    """Measured on the built world: the kind of entities, the fact-change process and the kinds of publishers that
    exist (with the official operators' publishing mode)."""
    mod = world.modules["infoeco"]
    kinds = sorted({s["kind"] + (f":{s['mode']}" if s["kind"] == "official" else "")
                    for _, s in world.state.scan("info_sources")})
    return {"entities": mod.cfg["kind"], "fact_change_process": mod.cfg["change"]["process"], "publishers": kinds}


CONFIGS = (
    DomainConfig("local_hazard", "train", _structure("local_hazard"),
                 "local businesses; facts change at random; official pages with lags, two news desks, a copying "
                 "aggregator and a content farm"),
    DomainConfig("software_releases", "train", _structure("software_releases"),
                 "software libraries; versions only move forward; every release is a new release-notes document; "
                 "static tutorials age; an aggregator copies the web"),
    DomainConfig("transit_announced", "train", _structure("transit_announced"),
                 "transit lines; timetable and fare changes are announced days ahead ('from day N'); a news desk "
                 "and a content farm"),
    DomainConfig("software_announced_spam", "eval_compositional", _structure("software_announced_spam"),
                 "software libraries (seen in training) with announced changes (seen in training) and content farms "
                 "(never combined with libraries in training)"),
    DomainConfig("local_wiki_shocks", "eval_structural", _structure("local_wiki_shocks"),
                 "local businesses with correlated regional shocks (a new change process) and a community wiki "
                 "corrected by readers who were misled (a new kind of publisher); no news desks"),
)


def build(config: DomainConfig, seed: int, overrides: dict) -> dict:
    ov = dict(overrides or {})
    rng = K.seeded_rng(config.name, seed)
    inst = generate_instance(rng, _merge(STRUCTURES[config.name], ov.pop("instance", {}) or {}))
    n = int(ov.pop("population", POPULATION))
    mods = K.order_modules([("identity", {}), ("bank", {}), ("admin", {}), ("artifacts", {}), ("cols", {}),
                            ("docs", {}), ("search", {}), ("pop", K.pop_config(n)), ("infoeco", inst)])
    return K.recipe(mods, {"max_actions_per_hour": 600})


def objective(world, principal: str, difficulty: int = 1) -> dict:
    d = max(1, int(difficulty))
    users = 15 + 10 * d
    text = (f"Run an information service on this world's web that people choose to rely on: over the last 10 of the "
            f"first 30 days, at least {users} different people must have looked things up through your service, "
            f"what they found and acted on must have been right at least 80% of the time and at least 10 points "
            f"more often than with the open web search, and the right answer must come near the top of what your "
            f"service returns.")
    return {"text": text, "description": text, "window_days": 10, "horizon_days": 30, "min_users": users,
            "min_success": 0.8, "margin": 0.1, "min_mrr": 0.6, "difficulty": d}


class InfoCurator(K.ReadingReference):
    """Scripted reference (evaluator side, tool-only): a fact-checking retrieval service. It indexes what the web
    search finds for every entity in the public directory, spends its daily inquiry calls on the facts its own users
    ask about most and has not checked recently, learns each publisher's accuracy from those checks, publishes a
    dated fact sheet per entity it has checked (pinned for that entity's queries, citing the documents that agreed)
    and ranks publishers by their measured accuracy, recent documents first."""
    cycle_s = 6 * HOUR

    def __init__(self, principal, objective):
        super().__init__(principal, objective)
        self.svc = None
        self.ents: list[dict] = []
        self.known: dict[str, str] = {}            # doc -> owner
        self.by_ent: dict[str, set] = {}
        self.indexed: set[str] = set()
        self.verified: dict[tuple, tuple] = {}     # (entity id, attr) -> (value, t)
        self.to_score: list[tuple] = []
        self.acc: dict[str, list[int]] = {}
        self.agree: dict[str, set] = {}
        self.cited: dict[str, list] = {}
        self.sheets: dict[str, str] = {}
        self.sheet_body: dict[str, str] = {}
        self.pending_sheets: set[str] = set()
        self.calls_day, self.calls = -1, 0

    def reads(self, t):
        if not self.ents:
            return [("ents", "info.entities", {})]
        out = [(f"s:{e['entity']}", "search.query", {"q": e["name"], "kind": "doc", "limit": 10}) for e in self.ents]
        if self.svc:
            out.append(("stats", "info.service_stats", {"service": self.svc}))
        need = sorted({d for eid, _, _ in self.to_score for d in self.by_ent.get(eid, ())})
        out += [(f"d:{d}", "docs.get", {"doc": d}) for d in need]
        return out

    def decide(self, t, got):
        if "ents" in got:
            self.ents = (got["ents"] or {}).get("entities", [])
        if self.svc is None:
            return [K.A("info.service_create", "svc", name="Checked Facts",
                        description="Opening times, prices, versions, departures and fares checked with the source "
                                    "and dated. Ask about any business, library or line.")]
        for e in self.ents:
            for x in (got.get(f"s:{e['entity']}") or {}).get("results", []):
                self.known.setdefault(x["id"], x["owner"])
                self.by_ent.setdefault(e["entity"], set()).add(x["id"])
        self._score(t, got)
        out = []
        new = [d for d in sorted(self.known) if d not in self.indexed]
        for k in range(0, len(new), 200):
            out.append(K.A("info.service_add", None, service=self.svc, docs=new[k:k + 200]))
        self.indexed |= set(new)
        words = dict((got.get("stats") or {}).get("top_query_words") or [])
        day = t // DAY
        if day != self.calls_day:
            self.calls_day, self.calls = day, 0
        cand = []
        for e in self.ents:
            dem = 1.0 + sum(words.get(w, 0) for w in name_tokens(e["name"]))
            for a in e["attributes"]:
                v = self.verified.get((e["entity"], a))
                age = (t - v[1]) / DAY if v else 60.0
                cand.append((-(dem * (1.0 + age)), e["entity"], a))
        for _, eid, a in sorted(cand)[: max(0, min(3, 12 - self.calls))]:
            out.append(K.A("info.inquire", f"verify:{eid}:{a}", entity=eid, attribute=a))
            self.calls += 1
        for e in self.ents:
            eid = e["entity"]
            vals = {a: self.verified[(eid, a)] for a in e["attributes"] if (eid, a) in self.verified}
            if not vals:
                continue
            body = " ".join(f"As of day {int(tt // DAY)}, " + _lc(ATTRS[a]["say"][0].format(n=e["name"], v=v))
                            for a, (v, tt) in sorted(vals.items()))
            body = f"{body} Checked directly with {e['name']}."
            title = f"{e['name']}: checked " + ", ".join(sorted(vals))
            if eid in self.sheets:
                if self.sheet_body.get(eid) != body:
                    self.sheet_body[eid] = body
                    out.append(K.A("docs.edit", None, doc=self.sheets[eid], title=title, body=body))
                agree = sorted(self.agree.get(eid, ()))[:50]
                if agree and agree != self.cited.get(eid):
                    self.cited[eid] = agree
                    out.append(K.A("info.cite", None, doc=self.sheets[eid], cites=agree))
            elif eid not in self.pending_sheets:
                self.pending_sheets.add(eid)
                self.sheet_body[eid] = body
                out.append(K.A("docs.create", f"sheet:{eid}", title=title, body=body, visibility="public"))
        mine = [d for d in self.sheets.values() if d not in self.indexed]
        if mine:
            out.append(K.A("info.service_add", None, service=self.svc, docs=mine))
            self.indexed |= set(mine)
        weights = {o: round(((a[0] + 1) / (a[1] + 2)) ** 2, 4) for o, a in self.acc.items() if a[1] >= 2}
        weights[self.me] = 4.0
        pins = [{"terms": " ".join(sorted(name_tokens(e["name"]))), "doc": self.sheets[e["entity"]]}
                for e in self.ents if e["entity"] in self.sheets]
        out.append(K.A("info.service_rank", None, service=self.svc, publisher_weights=weights,
                       recency_half_life_days=30, pins=pins[:300]))
        return out

    def _score(self, t, got):
        """Learn publishers' accuracy: compare what each document says with the values just checked."""
        day = t // DAY
        others = tuple(name_tokens(x["name"]) for x in self.ents)
        names = {e["entity"]: e["name"] for e in self.ents}
        for eid, attr, value in self.to_score:
            for did in sorted(self.by_ent.get(eid, ())):
                owner = self.known.get(did)
                d = got.get(f"d:{did}")
                if owner == self.me or not d:
                    continue
                c = read_claim(d["title"], d["body"], names[eid], attr, day, others)
                if c is None:
                    continue
                a = self.acc.setdefault(owner, [0, 0])
                a[0] += int(c[0] == value)
                a[1] += 1
                if c[0] == value:
                    self.agree.setdefault(eid, set()).add(did)
        self.to_score = []

    def handle(self, tag, result, action):
        if result is None:
            return
        if tag == "svc":
            self.svc = result["service"]
        elif tag and tag.startswith("verify:"):
            _, eid, attr = tag.split(":")
            self.verified[(eid, attr)] = (result["value"], self.t)
            self.to_score.append((eid, attr, result["value"]))
        elif tag and tag.startswith("sheet:"):
            self.sheets[tag.split(":", 1)[1]] = result["doc"]


class IndexEverything(K.ReadingReference):
    """Degenerate-strategy probe: a service that indexes everything the web search shows and curates nothing."""
    cycle_s = 12 * HOUR

    def __init__(self, principal, objective):
        super().__init__(principal, objective)
        self.svc = None
        self.ents: list[dict] = []
        self.indexed: set[str] = set()

    def reads(self, t):
        if not self.ents:
            return [("ents", "info.entities", {})]
        return [(f"s:{e['entity']}", "search.query", {"q": e["name"], "kind": "doc", "limit": 10}) for e in self.ents]

    def decide(self, t, got):
        if "ents" in got:
            self.ents = (got["ents"] or {}).get("entities", [])
        if self.svc is None:
            return [K.A("info.service_create", "svc", name="Everything Search",
                        description="Opening hours, prices, timetables, fares, software versions and documentation "
                                    "for every business, library and line.")]
        found = []
        for e in self.ents:
            for x in (got.get(f"s:{e['entity']}") or {}).get("results", []):
                if x["id"] not in self.indexed:
                    found.append(x["id"])
                    self.indexed.add(x["id"])
        return [K.A("info.service_add", None, service=self.svc, docs=found[k:k + 200])
                for k in range(0, len(found), 200)]

    def handle(self, tag, result, action):
        if tag == "svc" and result:
            self.svc = result["service"]


Reference = InfoCurator
IndexAllProbe = IndexEverything

DOMAIN = register(Domain(
    name="information", title="Information ecosystem", spec_part="9.11", required_mechanics=MECHANICS,
    configs=CONFIGS, build=build, objective=objective, evaluate=evaluate, reference=InfoCurator,
    mechanics=tuple(Mechanic(m, (), f"measured by information.mechanics_report: {m}",
                             count=K.measured(mechanics_report, m)) for m in MECHANICS),
    seats=(Seat("agent", 1_000_000, "Information service operator"),),
    actors=(Actor("population readers", ("cols.batch:info.lookup",), "individuals looking things up"),
            Actor("publishers", ("docs.create", "docs.edit"), "operators, news desks, aggregators, farms, wiki"),
            Actor("news desks", ("info.check",), "reporters checking facts with their own capacity")),
    degenerate=(Probe("index_everything", "a service that indexes every search result and curates nothing",
                      IndexEverything),),
    structure=structure, mode="B", horizon_days=30.0, slowest_effect_days=7.0, dt=3 * HOUR,
    reference_seeds=(0, 1), sources=("information.py", "_dom3.py"),
    doc=__doc__.split("\n\n")[0]))
