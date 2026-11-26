"""Event accounting and activity measures of a civilisation run (Gate P.5, P.6).

`Accountant` reads the event log as it grows (`feed(events)`) and checks, whenever asked, that what the events say
happened is what the state holds - each check an equality between an event-derived count and a state-derived one,
over the whole run since genesis:

    log            the event sequence has no gap and ends at the world's sequence number
    ledger         one `txns` record per `bank.txn` event; every balance equals the sum of its postings
                   (bank.check_ledger_consistency); money is conserved (bank invariants)
    sessions       sessions recorded in `pop.session` batches = the population's `n_sessions` column total
    goods          `make.retail_sold` events = the households' purchase counter (`goods.bought_n`) = units sold on
                   the retail listings (`sold_total`), and their cents = the households' spending (`goods.spent`)
    web purchases  purchases counted on web resources (`res.purchases`) = invoices the population paid to sites
    forum          posts in state = `forum.post` events
    energy         energy recorded in each region-day of grid use = production-run energy + other draws (each
                   addition is rounded to 1 Wh by the applier, so the tolerance is 0.5 Wh per recorded draw)

`activity(events)` classifies each event as economic (a decided payment between two participants; recurring
automatic charges such as hourly compute billing are not counted), social (posts, replies,
follows, votes, published content, messages, community acts, word of mouth) or neither; the long-run harness keeps
daily series of both.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

SYSTEM_ACCOUNTS = frozenset({"acct_genesis", "acct_households", "acct_external", "acct_fees"})
SOCIAL_TYPES = ("forum.post", "forum.reply", "forum.follow", "forum.vote", "content.published", "content.comment",
                "msg.", "community.", "social.follow", "social.share", "goods.wom", "creators.")


# recurring charges that run by themselves (hourly instance billing, upkeep, metered energy) and money entering from
# outside: real flows, but not evidence that anyone still trades
MECHANICAL_KINDS = frozenset({"compute", "opex", "energy", "income", "endowment", "grant"})


def is_economic(ev) -> bool:
    """A payment between two participants that someone decided to make (a purchase, an invoice paid, a trade, a
    licence, a wage...): not a recurring automatic charge and not money entering from outside the world."""
    if ev.type != "bank.txn" or ev.payload.get("kind") in MECHANICAL_KINDS:
        return False
    acc = [a for a, d in ev.payload["postings"] if a not in SYSTEM_ACCOUNTS]
    return len(acc) >= 2


def is_social(ev) -> bool:
    return ev.type.startswith(SOCIAL_TYPES)


def activity(events) -> dict:
    eco = soc = 0
    cents = 0
    for e in events:
        if is_economic(e):
            eco += 1
            cents += sum(d for _, d in e.payload["postings"] if d > 0)
        elif is_social(e):
            soc += 1
    return {"economic": eco, "economic_cents": cents, "social": soc}


class Accountant:
    def __init__(self, world):
        self.world = world
        self.last_seq = 0
        self.gaps = 0
        self.c = Counter()
        self.sessions = 0
        self.retail_units = 0
        self.retail_cents = 0
        self.energy = defaultdict(float)          # region|day -> kwh from events
        self.energy_n = Counter()
        self.posts = 0
        self.deleted = set()
        self.batch_rows_bad = 0
        self.pop_invoice_paid = 0

    def feed(self, events) -> None:
        st = self.world.state
        for e in events:
            if e.seq != self.last_seq + 1:
                self.gaps += 1
            self.last_seq = e.seq
            t = e.type
            self.c[t] += 1
            p = e.payload
            if t == "cols.batch":
                if p["kind"] == "pop.session":
                    self.sessions += int((p.get("extra") or {}).get("sessions", 0))
            elif t == "make.retail_sold":
                self.retail_units += int(p["qty"])
                self.retail_cents += int(p["cents"])
            elif t == "make.run":
                m = st.get("machines", p["machine"])
                f = st.get("facilities", m["facility"]) if m else None
                if f:
                    k = f"{f['region']}|{e.t // 86400}"
                    self.energy[k] += float(p["kwh"])
                    self.energy_n[k] += 1
            elif t == "make.energy":
                k = f"{p['region']}|{e.t // 86400}"
                self.energy[k] += float(p["kwh"])
                self.energy_n[k] += 1
            elif t == "forum.post":
                self.posts += 1
            elif t == "bank.invoice_paid":
                inv = st.get("invoices", p["id"])
                from ..substrate import ids
                if inv and ids.index(inv["payer"]) is not None:
                    self.pop_invoice_paid += 1

    def check(self) -> dict[str, list[str]]:
        w = self.world
        st = w.state
        out: dict[str, list[str]] = {}
        out["log"] = ([f"{self.gaps} gaps in the event sequence"] if self.gaps else []) + \
            ([f"accounted up to {self.last_seq}, world at {w.seq}"] if self.last_seq != w.seq else [])
        bank = w.modules.get("bank")
        if bank is not None:
            errs = []
            if st.count("txns") != self.c["bank.txn"]:
                errs.append(f"{st.count('txns')} ledger transactions, {self.c['bank.txn']} bank.txn events")
            errs += bank.check_ledger_consistency(w)[:5]
            errs += bank.check_invariants(w)[:5]
            out["ledger"] = errs
        from ..substrate import columns
        if "pop" in columns.tables(w):
            tot = int(columns.col(w, "pop", "n_sessions").astype(np.int64).sum())
            out["sessions"] = [] if tot == self.sessions else [f"n_sessions total {tot}, pop.session batches "
                                                                f"{self.sessions}"]
        if "goods" in columns.tables(w):
            b = int(columns.col(w, "goods", "bought_n").astype(np.int64).sum())
            sp = int(columns.col(w, "goods", "spent").astype(np.int64).sum())
            sold = sum(int(r.get("sold_total", 0)) for _, r in st.scan("retail"))
            errs = []
            if not (b == self.retail_units == sold):
                errs.append(f"goods purchases {b}, retail_sold units {self.retail_units}, listing sold_total {sold}")
            if sp != self.retail_cents:
                errs.append(f"goods spending {sp}, retail_sold cents {self.retail_cents}")
            out["goods"] = errs
        if "res" in columns.tables(w):
            rp = int(columns.col(w, "res", "purchases").astype(np.int64).sum())
            out["web_purchases"] = [] if rp == self.pop_invoice_paid else [
                f"web resources count {rp} purchases, the population paid {self.pop_invoice_paid} site invoices"]
        if "forum_posts" in st.tables:
            n = st.count("forum_posts")
            out["forum"] = [] if n == self.posts else [f"{n} posts in state, {self.posts} forum.post events"]
        if "grid_days" in st.tables:
            errs = []
            for k, g in st.scan("grid_days"):
                if abs(float(g["used"]) - self.energy.get(k, 0.0)) > 0.0005 * self.energy_n.get(k, 0) + 1e-6:
                    errs.append(f"grid {k}: used {g['used']}, events {round(self.energy.get(k, 0.0), 3)}")
            out["energy"] = errs[:5]
        return out

    def mismatches(self) -> int:
        return sum(len(v) for v in self.check().values())

    # ------------------------------------------------------------------ checkpoints (crucible.civ.longrun)
    _FIELDS = ("last_seq", "gaps", "sessions", "retail_units", "retail_cents", "posts", "batch_rows_bad",
               "pop_invoice_paid")

    def state(self) -> dict:
        """JSON-native state, so a long run interrupted after a checkpoint resumes its accounting exactly."""
        return {**{k: getattr(self, k) for k in self._FIELDS}, "c": dict(self.c), "energy": dict(self.energy),
                "energy_n": dict(self.energy_n), "deleted": sorted(self.deleted)}

    def restore(self, d: dict) -> "Accountant":
        for k in self._FIELDS:
            setattr(self, k, d[k])
        self.c = Counter(d["c"])
        self.energy = defaultdict(float, {k: float(v) for k, v in d["energy"].items()})
        self.energy_n = Counter(d["energy_n"])
        self.deleted = set(d["deleted"])
        return self
