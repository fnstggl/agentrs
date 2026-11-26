"""What one participant knows: a learner-private memory built only from its own observations and action results.

A self-play learner acts in a world of hundreds of tools whose arguments name things (repositories, products,
deals, organisations, people, packages, listings...). It can only name what it has seen. `Knowledge` is that
memory, kept separately by every participant (never shared, never filled by the harness):

* `by_field[name]` - short values seen under a field name anywhere in what the participant received (tool results,
  visible event payloads, entities, numbers): ``"repo" -> {"ann/textkit": t}``. Tool arguments are mostly named like
  the fields of the results and events that mention the same things, so this is how an argument slot learns its
  candidate values; nothing in it is specific to one tool or module;
* `ids` - every id-like value seen (with its kind, first/last time seen, how often, and whether it is the
  participant's own: returned by its own state-changing action or related to it in its entities);
* `words` - topic words of the texts it read (for the titles and descriptions it writes);
* `accounts`, `orgs` - its own bank accounts and organisations (from its numbers and entities);
* `tool_stats` - its own record of each tool: tries, successes, last error (learner-side experience, e.g. for a
  learner's own novelty or competence incentives).

Everything here is derived from `Observation` dicts and action-result dicts the participant itself received
(`crucible.experience`), so it holds no hidden state of the world or of anybody else.
"""
from __future__ import annotations

import re
from collections import Counter, OrderedDict
from typing import Any

from ..experience.features import id_kind, is_id

TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,63}$")
WORD_RE = re.compile(r"[a-z]{4,12}")
CAP_RE = re.compile(r"\b([a-z][a-z_]{1,15}\.[a-z][a-z_]{1,24})\b")
STOP = frozenset("""about above after again against also among another around because been before being below
between both could does doing down during each from further have having here into itself just more most none only
other over same should some such than that their them then there these they this those through under until very
were what when where which while with would your yours public private their every after before using used uses
none true false null with your into from that this have will been more less cents seconds""".split())
MAX_FIELD_VALUES = 48
MAX_IDS = 4000
MAX_WORDS = 400


class Knowledge:
    """A participant's private memory of the ids, names and words it has seen (see the module docstring)."""

    def __init__(self, me: str):
        self.me = me
        self.handle = re.sub(r"[^a-z0-9]", "", me.split(":", 1)[-1].lower()) or "agent"
        self.by_field: dict[str, OrderedDict] = {}
        self.ids: OrderedDict[str, dict] = OrderedDict()
        self.words: Counter = Counter()
        self.accounts: list[str] = []
        self.orgs: list[str] = []
        self.mine: set[str] = set()
        self.tool_stats: dict[str, dict] = {}
        self.packages: OrderedDict[str, dict] = OrderedDict()   # import path -> {package, api}: packages it found
        self.caps: list[str] = []
        self.fresh = 0                       # counter for names it invents (repositories, handles, hosts)
        self.t = 0
        self.balances: dict[str, int] = {}

    # ------------------------------------------------------------------ intake
    def note_value(self, field: str, value: Any, t: int, mine: bool = False) -> None:
        if not isinstance(value, str) or not TOKEN_RE.match(value) or value == self.me:
            return
        d = self.by_field.setdefault(field, OrderedDict())
        d.pop(value, None)
        d[value] = int(t)
        while len(d) > MAX_FIELD_VALUES:
            d.popitem(last=False)
        if is_id(value) or "/" in value or "@" in value:
            self.note_id(value, t, field, mine)

    def note_id(self, value: str, t: int, field: str = "", mine: bool = False) -> None:
        rec = self.ids.get(value)
        if rec is None:
            rec = {"kind": id_kind(value), "first": int(t), "last": int(t), "n": 0, "mine": False, "fields": set()}
            self.ids[value] = rec
        else:
            self.ids.move_to_end(value)
        rec["last"] = int(t)
        rec["n"] += 1
        if field:
            rec["fields"].add(field)
        if mine:
            rec["mine"] = True
            self.mine.add(value)
        while len(self.ids) > MAX_IDS:
            old, r = self.ids.popitem(last=False)
            self.mine.discard(old)

    def harvest(self, x: Any, t: int, mine: bool = False, key: str = "", depth: int = 0) -> None:
        """Every short value in a JSON-like value, under the field name it appeared with."""
        if depth > 8:
            return
        if isinstance(x, dict):
            imp, api = x.get("import"), x.get("api")
            if isinstance(imp, str) and isinstance(api, dict) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,80}", imp):
                self.packages.pop(imp, None)
                self.packages[imp] = {"package": x.get("package"), "api": [str(a) for a in list(api)[:8]],
                                      "owner": x.get("owner")}
                while len(self.packages) > 16:
                    self.packages.popitem(last=False)
            for k, v in x.items():
                k = str(k)
                if isinstance(v, (dict, list)):
                    self.harvest(v, t, mine, k, depth + 1)
                elif isinstance(v, str):
                    self.note_value(k, v, t, mine)
                    if len(v) > 24 or " " in v:
                        self.read_text(v)
                if isinstance(k, str) and (is_id(k) or k.startswith(("o:", "p:"))):
                    self.note_id(k, t, key, mine)
        elif isinstance(x, list):
            for v in x[:200]:
                if isinstance(v, str):
                    self.note_value(key or "item", v, t, mine)
                else:
                    self.harvest(v, t, mine, key, depth + 1)

    def read_text(self, text: str) -> None:
        for w in WORD_RE.findall(text.lower()):
            if w not in STOP:
                self.words[w] += 1
        if len(self.words) > MAX_WORDS:
            self.words = Counter(dict(self.words.most_common(MAX_WORDS // 2)))

    def ingest_observation(self, obs: dict) -> None:
        key = (int(obs.get("t", 0)), int(obs.get("cursor", -1)), id(obs))
        if key == getattr(self, "_last_obs", None):
            return                          # the same observation handed over twice (observe, then act)
        self._last_obs = key
        t = int(obs.get("t", 0))
        self.t = t
        bal = {}
        for k, v in (obs.get("numbers") or {}).items():
            if k.startswith("bank.balances."):
                acct = k.rsplit(".", 1)[-1]
                bal[acct] = int(v)
                if acct not in self.accounts:
                    self.accounts.append(acct)
                self.note_id(acct, t, "account", mine=True)
        self.balances = bal
        for ent in obs.get("entities") or []:
            eid = ent.get("id")
            rels = ent.get("relations") or []
            if eid == self.me:
                # the organisations it runs: its own entity's membership relations (nothing else makes one "its")
                for r in rels:
                    o = r.get("to")
                    if r.get("rel") == "member_or_admin_of" and isinstance(o, str) and o.startswith("o:"):
                        self.note_id(o, t, "org", mine=True)
                        if o not in self.orgs:
                            self.orgs.append(o)
            elif eid:
                self.note_id(eid, t, ent.get("kind", ""))
        for e in obs.get("events") or []:
            own = e.get("actor") == self.me or bool(e.get("cause"))
            self.harvest(e.get("payload") or {}, t, mine=False)
            typ = str(e.get("type", ""))
            pay = e.get("payload") or {}
            if isinstance(pay, dict) and isinstance(pay.get("id"), str):
                # an event's own `id` names the thing of its type ("deal.offer" -> a deal): file it under that noun
                self.note_value(typ.split(".", 1)[0], pay["id"], t, mine=own and e.get("actor") == self.me)
            actor = e.get("actor")
            if isinstance(actor, str) and actor != self.me and actor.startswith(("p:", "o:")):
                self.note_value("actor", actor, t)
        for line in (obs.get("text") or [])[:60]:
            self.read_text(str(line))
        if not self.caps:
            caps: list[str] = []
            for tool in obs.get("tools") or []:
                for c in CAP_RE.findall(str(tool.get("description", ""))):
                    if c not in caps:
                        caps.append(c)
            self.caps = caps[:40]

    def ingest_result(self, action: dict | None, result: dict | None, read_only: bool) -> None:
        if not action or result is None:
            return
        tool = str(action.get("tool", ""))
        st = self.tool_stats.setdefault(tool, {"tries": 0, "ok": 0, "last_error": None, "last_t": None})
        st["tries"] += 1
        st["last_t"] = self.t
        if result.get("ok"):
            st["ok"] += 1
            st["last_error"] = None
            before = set(self.ids)
            self.harvest(result.get("result"), self.t)
            if not read_only:
                # what its own state-changing action brought into existence (ids it had never seen) is its own
                for v in set(self.ids) - before:
                    if v.startswith(("p:", "o:", "svc:")):
                        continue                # principals are its own only by membership (entities)
                    self.ids[v]["mine"] = True
                    self.mine.add(v)
        else:
            err = result.get("error") or {}
            st["last_error"] = err.get("code") if isinstance(err, dict) else str(err)[:40]

    # ------------------------------------------------------------------ queries
    def values(self, field: str) -> list[str]:
        return list(reversed(self.by_field.get(field, {}).keys()))

    def ids_of(self, kinds: tuple[str, ...], mine: bool | None = None, limit: int = 12) -> list[str]:
        out = []
        for v in reversed(self.ids):
            r = self.ids[v]
            if r["kind"] in kinds and (mine is None or r["mine"] == mine) and v != self.me:
                out.append(v)
                if len(out) >= limit:
                    break
        return out

    def topic_words(self, k: int = 4) -> list[str]:
        return [w for w, _ in self.words.most_common(k)]

    def fresh_name(self, stem: str = "") -> str:
        self.fresh += 1
        return f"{self.handle[:12]}{stem[:8]}{self.fresh}"

    def counts(self) -> dict[str, int]:
        c: Counter = Counter()
        m: Counter = Counter()
        for r in self.ids.values():
            c[r["kind"]] += 1
            if r["mine"]:
                m[r["kind"]] += 1
        return {"known": dict(c), "mine": dict(m)}
