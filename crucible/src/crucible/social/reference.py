"""Scripted reference participants for the social internet (MASTER_SPEC Clarification 1: environment capabilities are
validated with scripted reference agents, besides the population and learned participants).

A `ReferenceParticipant` is an ordinary principal (`p:<handle>`) that acts ONLY through the world's tools, with the
same permissions, costs and visibility as any agent. It has private interests (a set of words), a sociability, an
attention budget (items it can open per day) and a running satisfaction per community. Each simulated day it

1. collects candidates from the discovery channels it can see: its chronological timeline (`social.timeline`),
   its communities' feeds (`community.feed`), direct shares in its conversations (`msg.conversations` /
   `msg.history`), newsletter issues in its mailbox (`mail.inbox` / `mail.read`), search for one of its interests
   (`search.query`), the day's product launches (`launch.today`) and a platform's latest items (`<site>.latest`);
2. opens candidates within its attention budget (`content.open`), judges each by the overlap between the words of
   the item and its own interests (it reads the actual artifact text, never a score), and then - by seeded draws
   from its own sociability - comments, shares the item with a contact whose interests it knows, or follows a
   creator whose items it found relevant twice;
3. takes part in communities: visits those it belongs to, posts in them when it is a creator, leaves a community
   whose recent posts keep disappointing it, joins communities from the directory whose subject matches its
   interests, and - if none matches its main interest and it is sociable enough - founds one, writing the subject
   in its own words;
4. as a creator, publishes original text artifacts about its interests on platforms (made with `studio.write`).

Every decision is a seeded draw from the participant's own state; the world records each as an ordinary command.
These participants exist to exercise and measure the social mechanisms (gates C.2, C.4, C.5); the population's
own behaviour is workstream B's.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from ..core.kernel import World

TOPICS: dict[str, list[str]] = {
    "sailing": ["sailing", "boat", "wind", "harbour", "knots", "regatta", "mast", "keel", "tide", "anchor"],
    "astronomy": ["telescope", "galaxy", "nebula", "orbit", "eclipse", "comet", "stargazing", "planet", "lens",
                  "observatory"],
    "baking": ["sourdough", "flour", "oven", "crust", "yeast", "dough", "pastry", "bread", "starter", "proofing"],
    "gardening": ["compost", "seedlings", "soil", "tomatoes", "pruning", "mulch", "greenhouse", "roses", "harvest",
                  "watering"],
    "chess": ["opening", "endgame", "gambit", "rook", "bishop", "tactics", "checkmate", "pawn", "tournament",
              "castling"],
    "cycling": ["bicycle", "gears", "climb", "saddle", "pedals", "touring", "puncture", "helmet", "peloton",
                "cadence"],
    "coding": ["python", "compiler", "debugging", "functions", "testing", "refactoring", "library", "algorithm",
               "recursion", "types"],
    "birding": ["warbler", "binoculars", "migration", "nest", "heron", "songbird", "feathers", "plumage", "owl",
                "hawk"],
}
FILLER = ["about", "today", "notes", "my", "the", "and", "with", "on", "for", "a", "new", "some", "great", "this",
          "week", "tips", "first", "best"]
TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> set[str]:
    return set(TOKEN.findall(str(text).lower()))


@dataclass
class ReferenceParticipant:
    world: World
    principal: str
    handle: str
    interests: list[str]                    # private: topic names
    sociability: float
    creator: bool
    seed: int
    attention: int = 8                       # items it can open per day
    max_communities: int = 3
    seen: set = field(default_factory=set)
    liked_creators: dict = field(default_factory=dict)
    contacts: dict = field(default_factory=dict)          # principal -> topics it knows they like
    comm_sat: dict = field(default_factory=dict)          # community -> running satisfaction
    log: list = field(default_factory=list)               # (day, channel, item, relevant)
    founded: list = field(default_factory=list)
    mail_seen: set = field(default_factory=set)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.words = sorted({w for t in self.interests for w in TOPICS[t]})

    # ------------------------------------------------------------------ tools
    def call(self, tool: str, **args):
        r = self.world.execute(self.principal, {"tool": tool, "args": args})
        return r.result if r.ok else None

    def relevance(self, text: str) -> float:
        return len(tokens(text) & set(self.words)) / 3.0

    def compose(self, topic: str, n: int = 40) -> tuple[str, str]:
        words = TOPICS[topic]
        body = []
        for _ in range(n):
            body.append(words[int(self.rng.integers(len(words)))] if self.rng.random() < 0.45
                        else FILLER[int(self.rng.integers(len(FILLER)))])
        title = f"{words[int(self.rng.integers(len(words)))].capitalize()} {FILLER[int(self.rng.integers(len(FILLER)))]}" \
                f" {words[int(self.rng.integers(len(words)))]}"
        return title, " ".join(body)

    # ------------------------------------------------------------------ discovery
    def candidates(self, day: int) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        tl = self.call("social.timeline", limit=20)
        for e in (tl or {}).get("entries", []):
            out.append(("following", e["item"]["id"]))
        for c in self.call("community.mine") or []:
            if c["your_status"] == "member":
                f = self.call("community.feed", community=c["community"], limit=10)
                for it in (f or {}).get("items", []):
                    out.append(("community", it["id"]))
        for conv in self.call("msg.conversations") or []:
            if conv["unread"]:
                h = self.call("msg.read", conversation=conv["conversation"], limit=20)
                for m in (h or {}).get("messages", []):
                    if m.get("item") and m["sender"] != self.principal:
                        out.append(("share", m["item"]))
        for m in self.call("mail.inbox", unread_only=True, limit=20) or []:
            if m["msg"] in self.mail_seen:
                continue
            self.mail_seen.add(m["msg"])
            full = self.call("mail.read", msg=m["msg"])
            hit = re.search(r"content (c\d+)", (full or {}).get("body", ""))
            if hit:
                out.append(("newsletter", hit.group(1)))
        q = self.words[int(self.rng.integers(len(self.words)))]
        for r in (self.call("search.query", q=q, limit=5) or {}).get("results", []):
            if re.fullmatch(r"c\d+", str(r.get("id"))):
                out.append(("search", r["id"]))
        for L in (self.call("launch.today") or {}).get("launches", [])[:5]:
            out.append(("launch", L["id"]))
        site = ["posts", "blog", "pics"][int(self.rng.integers(3))]
        for it in (self.call(f"{site}.latest", limit=5) or {}).get("items", []):
            out.append(("latest", it["id"]))
        return out

    def browse(self, day: int) -> None:
        budget = self.attention
        order = {"share": 0, "newsletter": 1, "following": 2, "community": 3, "search": 4, "launch": 5, "latest": 6}
        cand = sorted(self.candidates(day), key=lambda x: order[x[0]])
        for channel, cid in cand:
            if budget <= 0:
                break
            if cid in self.seen:
                continue
            item = self.call("content.open", id=cid)
            if item is None:
                continue
            budget -= 1
            self.seen.add(cid)
            rel = self.relevance(" ".join([item["title"], item.get("description", ""), " ".join(item["tags"]),  # not-a-content-score: the scripted reader's own judgement computed from the text it read
                                           item.get("text", "")]))
            relevant = rel >= 1.0
            self.log.append((day, channel, cid, relevant))
            if item.get("community"):
                c = item["community"]
                self.comm_sat[c] = 0.7 * self.comm_sat.get(c, 0.5) + 0.3 * (1.0 if relevant else 0.0)
            if not relevant or item["creator"] == self.principal:
                continue
            cr = item["creator"]
            self.liked_creators[cr] = self.liked_creators.get(cr, 0) + 1
            if self.liked_creators[cr] >= 2 and self.rng.random() < 0.5 + 0.4 * self.sociability:
                self.call("social.follow", target=cr, via="content")
            if self.rng.random() < 0.25 * self.sociability:
                tool = "community.comment" if item.get("community") else "content.comment"
                self.call(tool, id=cid, text=f"Thanks, this helps with {self.words[int(self.rng.integers(len(self.words)))]}.")
            fits = [p for p, ts in sorted(self.contacts.items()) if tokens(" ".join(ts)) & tokens(item.get("text", ""))]
            if fits and self.rng.random() < 0.3 * self.sociability:
                to = fits[int(self.rng.integers(len(fits)))]
                self.call("social.share", id=cid, to=[to], note="you might like this")

    # ------------------------------------------------------------------ communities
    def communities(self, day: int) -> None:
        mine = [c for c in (self.call("community.mine") or []) if c["your_status"] == "member"]
        for c in mine:
            cid = c["community"]
            self.call("community.visit", community=cid)
            if self.comm_sat.get(cid, 0.5) < 0.25 and c["role"] != "owner":
                self.call("community.leave", community=cid)
                self.comm_sat.pop(cid, None)
            elif self.creator and self.rng.random() < 0.5:
                topic = self.interests[int(self.rng.integers(len(self.interests)))]
                title, body = self.compose(topic)
                a = self.call("studio.write", format="text", title=title, body=body)
                if a:
                    self.call("community.post", community=cid, ref=a["ref"], title=title, tags=[topic])
        if len(mine) >= self.max_communities:
            return
        have = {c["community"] for c in mine}
        best, best_score = None, 0
        for c in self.call("community.directory", limit=50) or []:
            if c["community"] in have or c["join"] != "open":
                continue
            s = len(tokens(c["subject"] + " " + " ".join(c["tags"])) & set(self.words))
            if s > best_score:
                best, best_score = c["community"], s
        if best is not None and best_score >= 1 and self.rng.random() < 0.3 + 0.5 * self.sociability:
            self.call("community.join", community=best)
            return
        main = self.interests[0]
        if best is None and not self.founded and self.sociability > 0.6 and self.rng.random() < 0.5:
            w = TOPICS[main]
            picks = [w[int(i)] for i in self.rng.choice(len(w), 3, replace=False)]
            slug = f"{picks[0]}-{self.handle}"[:40]
            subject = f"People who care about {picks[0]}, {picks[1]} and {picks[2]}"
            if self.call("community.found", id=slug, name=f"{picks[0].capitalize()} circle", subject=subject,
                         tags=[main]):
                self.founded.append(slug)

    # ------------------------------------------------------------------ creating
    def create(self, day: int) -> None:
        if not self.creator or self.rng.random() > 0.6:
            return
        topic = self.interests[int(self.rng.integers(len(self.interests)))]
        title, body = self.compose(topic)
        site = ["posts", "blog"][int(self.rng.integers(2))]
        fmt = "text" if site == "posts" else "html"
        a = self.call("studio.write", format=fmt, title=title, body=body)
        if a:
            self.call(f"{site}.publish", ref=a["ref"], title=title, tags=[topic])

    def newsletters(self, day: int) -> None:
        """Sociable creators start a newsletter and send an issue a week; readers subscribe to newsletters whose
        name matches their interests (issues then reach them by mail)."""
        if self.creator and self.sociability > 0.5:
            mine = [n for n in (self.call("letters.list", limit=100) or []) if n["owner"] == self.principal]
            if not mine:
                topic = self.interests[0]
                self.call("letters.create", name=f"{topic.capitalize()} weekly by {self.handle}",
                          description=f"news about {' '.join(TOPICS[topic][:5])}")
            elif day % 7 == 3:
                topic = self.interests[0]
                title, body = self.compose(topic, 60)
                a = self.call("studio.write", format="html", title=title, body=body)
                if a:
                    self.call("letters.send", newsletter=mine[0]["newsletter"], ref=a["ref"], title=title,
                              tags=[topic])
        if day % 5 == 0:
            for n in self.call("letters.list", limit=100) or []:
                if n["owner"] != self.principal and tokens(n["name"]) & set(self.interests) and \
                        self.rng.random() < 0.3 + 0.4 * self.sociability:
                    self.call("letters.subscribe", newsletter=n["newsletter"])

    def day(self, day: int) -> None:
        self.create(day)
        self.newsletters(day)
        self.browse(day)
        self.communities(day)


def make_participants(world: World, n: int, *, seed: int = 0, creators_share: float = 0.3,
                      endowment: int = 50_000, prefix: str = "ref") -> list[ReferenceParticipant]:
    """Admit `n` reference participants (operator tool `admin.join`) with seeded private interests."""
    rng = np.random.default_rng(seed)
    names = sorted(TOPICS)
    out = []
    for i in range(n):
        h = f"{prefix}{i}"
        p = world.system_call("admin.join", {"handle": h, "endowment": endowment, "kind": "npc"})["principal"]
        k = 1 + int(rng.random() < 0.5)
        ints = [names[int(j)] for j in rng.choice(len(names), k, replace=False)]
        out.append(ReferenceParticipant(world, p, h, ints, float(rng.random()), bool(rng.random() < creators_share),
                                        seed * 1000 + i))
    # each participant knows the interests of a few others (people it would share things with)
    for rp in out:
        for j in rng.choice(n, min(4, n - 1), replace=False):
            other = out[int(j)]
            if other is not rp:
                rp.contacts[other.principal] = other.words
    return out


def run_days(world: World, participants: list[ReferenceParticipant], days: int, start_day: int | None = None,
             hook=None) -> None:
    """Advance the world day by day; each participant takes its day (in a seeded order) at a time of its own."""
    DAY = 86400
    d0 = world.now // DAY if start_day is None else start_day
    rng = np.random.default_rng(len(participants) * 7919 + d0)
    for d in range(d0, d0 + days):
        order = rng.permutation(len(participants))
        slots = np.sort(rng.integers(8 * 3600, 22 * 3600, len(participants)))
        for k, j in enumerate(order):
            t = d * DAY + int(slots[k])
            if t > world.now:
                world.advance_time(t)
            participants[int(j)].day(d)
        if hook is not None:
            hook(d)
        world.advance_time((d + 1) * DAY)
