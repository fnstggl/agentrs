"""Web search (host `search`): the discovery layer of the synthetic internet.

Every public artefact that should be findable (forum posts, public docs, market listings, company pages, job
ads, deployed services, public repositories) is indexed here by the module that owns it, through the internal
API `index(ctx, doc_id, kind, title, text, owner, ref=None, hosts=())` / `unindex(ctx, doc_id)`. A document
carries the hostnames it references; whoever sees it in a result *learns* those hosts (`World.learn`), which is
how agents discover services and tools they did not know about.

Ranking is BM25 over lowercase word tokens (title tokens weigh double), multiplied by an optional quality
signal `world.hooks["rank_signal"](world, doc_record) -> float` (reputation, uptime, ... supplied by other
modules; a signal <= 0 delists a document), with a recency tie-break (index time, then index order). Sponsored results come from
`world.hooks["sponsored"](world, q, kind, limit)` (the ads module) and are placed first, labelled.

Determinism: the inverted index is a runtime cache (`world.runtime["search.index"]`) derived from the
`search_docs` table. Appliers only mark documents dirty; the cache reconciles dirty documents against the
authoritative records (by record identity) before every query, and keeps re-checking documents touched inside
the current transaction until it ends, so rollbacks (whole-command or savepoint) can never leave it stale. All
maintained statistics are integers and scores are accumulated in sorted term order, so results are identical
to a from-scratch computation (checked by `check_invariants`).
"""
from __future__ import annotations

import heapq
import math
import re
from types import MappingProxyType
from typing import Iterable

from ..core import schema as S
from ..core.errors import NotFound
from ..core.kernel import Ctx, Module, applier, endpoint, register_module

TOKEN = re.compile(r"[a-z0-9]+")
HOST_RE = re.compile(r"([a-z0-9][a-z0-9-]{1,40})\.crucible")
HOST_NAME = r"[a-z0-9][a-z0-9-]{1,40}"
KIND_RE = r"[a-z][a-z0-9_]{0,23}"
KINDS = ("post", "doc", "listing", "company", "job", "service", "repo")
MAX_TITLE = 200
MAX_TEXT = 8000
MAX_HOSTS = 16
MAX_SPONSORED = 3
SNIPPET = 240
K1, B = 1.2, 0.75
TITLE_W = 2
RUNTIME_KEY = "search.index"


# ====================================================================== helpers shared with other modules
def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric word tokens (the same tokenizer is used for documents, queries and ad keywords)."""
    return [t for t in TOKEN.findall(str(text).lower()) if len(t) <= 40]


def is_host(world, name: str) -> bool:
    """True if `name` is a service host: a built-in host, or one the `is_host(state, name)` hook accepts
    (agent-deployed services are registered by the cloud/hosting module through that hook)."""
    hook = world.hooks.get("is_host")
    if hook is not None and hook(world.state, name):
        return True
    return any(e.host == name for e in world.endpoints.values())


def extract_hosts(world, text: str, extra: Iterable[str] = ()) -> list[str]:
    """Hostnames referenced in free text as `<name>.crucible` that are real hosts, plus `extra` (trusted hosts
    supplied by internal callers). Sorted, unique, at most MAX_HOSTS."""
    found = {str(h)[:64] for h in extra if h}
    for name in sorted(set(HOST_RE.findall(str(text).lower()))):
        if name not in found and is_host(world, name):
            found.add(name)
    return sorted(found)[:MAX_HOSTS]


def search_module(world) -> "Search | None":
    return world.modules.get("search")  # type: ignore[return-value]


def index(ctx: Ctx, doc_id: str, kind: str, title: str, text: str, owner: str, ref: str | None = None,
          hosts: Iterable[str] = ()) -> None:
    """Module-level convenience: index a document if the search module is installed (no-op otherwise)."""
    mod = search_module(ctx.world)
    if mod is not None:
        mod.index(ctx, doc_id, kind, title, text, owner, ref, hosts)


def unindex(ctx: Ctx, doc_id: str) -> None:
    mod = search_module(ctx.world)
    if mod is not None:
        mod.unindex(ctx, doc_id)


def query(world, q: str, kind: str | None = None, limit: int = 10) -> list[dict]:
    """Pure ranked search over the current state: sponsored results first (labelled), then organic results."""
    mod = search_module(world)
    if mod is None:
        return []
    sponsored, organic, _ = mod.search(world, q, kind, limit)
    return sponsored + organic


# ====================================================================== the runtime inverted index
def _term_freqs(rec) -> dict[str, int]:
    tf: dict[str, int] = {}
    for t in tokenize(rec["title"]):
        tf[t] = tf.get(t, 0) + TITLE_W
    for t in tokenize(rec["text"]):
        tf[t] = tf.get(t, 0) + 1
    for h in rec["hosts"]:
        for t in tokenize(h):
            tf[t] = tf.get(t, 0) + 1
    return tf


class _Index:
    """Inverted index over `search_docs`, reconciled lazily against the authoritative records."""

    def __init__(self, state):
        self.state = state
        self.table = state.tables["search_docs"]
        self.recs: dict[str, dict] = {}
        self.tfs: dict[str, dict[str, int]] = {}
        self.dl: dict[str, int] = {}
        self.post: dict[str, dict[str, int]] = {}
        self.total = 0
        self.dirty: set[str] = set()
        self.journal = getattr(state, "_journal", None)
        for did in sorted(self.table):
            self._add(did, self.table[did])
        # built inside a transaction: anything may still roll back, so keep re-checking everything until it ends
        self.prov: set[str] = set(self.table) if self.journal is not None else set()

    def _add(self, did: str, rec: dict) -> None:
        tf = _term_freqs(rec)
        n = sum(tf.values())
        self.recs[did] = rec
        self.tfs[did] = tf
        self.dl[did] = n
        self.total += n
        for t, c in tf.items():
            self.post.setdefault(t, {})[did] = c

    def _remove(self, did: str) -> None:
        if self.recs.pop(did, None) is None:
            return
        self.total -= self.dl.pop(did)
        for t in self.tfs.pop(did):
            p = self.post[t]
            del p[did]
            if not p:
                del self.post[t]

    def sync(self) -> None:
        if not hasattr(self.state, "_journal"):
            # the store no longer exposes its journal: fall back to a full identity scan (always correct)
            check = set(self.table) | set(self.recs)
        else:
            check = (self.dirty | self.prov) if self.prov else self.dirty
        for did in sorted(check):
            cur = self.table.get(did)
            if self.recs.get(did) is not cur:
                self._remove(did)
                if cur is not None:
                    self._add(did, cur)
        j = getattr(self.state, "_journal", None)
        if j is None:
            self.prov = set()
        else:
            if j is not self.journal:
                self.prov = set()
            self.prov |= self.dirty
        self.dirty = set()
        self.journal = j

    def same_as(self, other: "_Index") -> bool:
        return (self.dl == other.dl and self.post == other.post and self.total == other.total
                and set(self.recs) == set(other.recs))


# ====================================================================== the module
@register_module
class Search(Module):
    name = "search"
    host = "search"

    def install(self, world):
        world.state.declare_table("search_docs", index=("kind", "owner"))

    def on_load(self, world):
        world.runtime.pop(RUNTIME_KEY, None)

    def _index(self, world) -> _Index:
        idx = world.runtime.get(RUNTIME_KEY)
        if idx is None or idx.state is not world.state or idx.table is not world.state.tables["search_docs"]:
            idx = _Index(world.state)
            world.runtime[RUNTIME_KEY] = idx
        else:
            idx.sync()
        return idx

    def _mark(self, did: str) -> None:
        idx = self.world.runtime.get(RUNTIME_KEY)
        if idx is not None:
            idx.dirty.add(did)

    # ------------------------------------------------------------------ internal API
    def index(self, ctx: Ctx, doc_id: str, kind: str, title: str, text: str, owner: str, ref: str | None = None,
              hosts: Iterable[str] = ()) -> None:
        """Insert or replace a public document. Callers are responsible for only indexing public content."""
        new = {"kind": str(kind)[:24], "title": str(title)[:MAX_TITLE], "text": str(text)[:MAX_TEXT],
               "owner": owner, "ref": ref, "hosts": sorted({str(h)[:64] for h in hosts if h})[:MAX_HOSTS]}
        old = ctx.state.get("search_docs", doc_id)
        if old is not None and all(old.get(k) == v for k, v in new.items()):
            return
        ctx.emit("search.index", {"id": doc_id, **new}, vis="internal")

    def unindex(self, ctx: Ctx, doc_id: str) -> None:
        if ctx.state.has("search_docs", doc_id):
            ctx.emit("search.unindex", {"id": doc_id}, vis="internal")

    def search(self, world, q: str, kind: str | None = None, limit: int = 10) -> tuple[list, list, int]:
        """(sponsored, organic, number of organic matches). Pure: reads state and hooks only."""
        limit = max(0, min(int(limit), 100))
        toks = sorted(set(tokenize(q)))
        sponsored: list[dict] = []
        hook = world.hooks.get("sponsored")
        if hook is not None and toks and limit:
            for ad in (hook(world, q, kind, min(limit, MAX_SPONSORED)) or [])[:MAX_SPONSORED]:
                sponsored.append(_ad_result(ad))
        organic, matched = self._organic(world, toks, kind, limit)
        return sponsored, organic, matched

    def _organic(self, world, toks: list[str], kind: str | None, limit: int) -> tuple[list, int]:
        st = world.state
        tbl = st.tables["search_docs"]
        if not limit:
            return [], 0
        if not toks:  # browse: most recent documents (of a kind)
            ids = st.find_ids("search_docs", "kind", kind) if kind else st.ids("search_docs")
            top = heapq.nsmallest(limit, ids, key=lambda i: (-tbl[i]["t"], -tbl[i]["seq"], i))
            return [_result(i, tbl[i], 0.0) for i in top], len(ids)
        idx = self._index(world)
        n_docs = len(idx.dl)
        avg = idx.total / n_docs if n_docs and idx.total else 1.0
        scores: dict[str, float] = {}
        for t in toks:
            posting = idx.post.get(t)
            if not posting:
                continue
            df = len(posting)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for did, tf in posting.items():
                if kind is not None and tbl[did]["kind"] != kind:
                    continue
                s = idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * idx.dl[did] / avg))
                scores[did] = scores.get(did, 0.0) + s
        sig = world.hooks.get("rank_signal")
        keyed = []
        for did in sorted(scores):
            sc = scores[did]
            rec = tbl[did]
            if sig is not None:
                m = float(sig(world, MappingProxyType(rec)))
                if not m > 0:          # also rejects NaN
                    continue
                sc *= m
            if sc > 0:
                keyed.append((-sc, -rec["t"], -rec["seq"], did))
        top = heapq.nsmallest(limit, keyed)
        return [_result(did, tbl[did], -neg) for neg, _, _, did in top], len(keyed)

    # ------------------------------------------------------------------ endpoints
    @endpoint("query", S.obj({"q": S.s_str("search terms", maxLength=200),
                              "kind": S.s_str("restrict to one document kind: " + ", ".join(KINDS), pattern=KIND_RE,
                                              maxLength=24),
                              "limit": S.s_int("number of organic results", minimum=1, maximum=20, default=10)},
                             ["q"]), cost_s=2, read_only=True)
    def ep_query(self, ctx, q, kind=None, limit=10):
        """Search the public web of this world (posts, docs, listings, companies, jobs, services, repos).
        Returns {results, matched}: sponsored results first (sponsored=true), then organic results ranked by
        relevance. Every result lists the service hosts it references; you learn those hosts, so their tools
        become available to you. An empty query lists the most recent documents."""
        sponsored, organic, matched = self.search(ctx.world, q, kind, limit)
        if sponsored:
            ads = ctx.world.modules.get("ads")
            camp = [r["campaign"] for r in sponsored if r.get("campaign")]
            if ads is not None and camp:
                charged = set(ads.record_impressions(ctx, camp, 1))  # type: ignore[attr-defined]
                sponsored = [r for r in sponsored if r.get("campaign") is None or r["campaign"] in charged]
        results = sponsored + organic
        for r in results:
            for h in r["hosts"]:
                ctx.world.learn(ctx, ctx.actor, "host", h)
        return {"results": results, "matched": matched}

    @endpoint("get", S.obj({"id": S.s_str("document id from a search result", maxLength=80)}, ["id"]), cost_s=2,
              read_only=True)
    def ep_get(self, ctx, id):
        """Full indexed text of one search result (you learn the hosts it references)."""
        rec = ctx.state.get("search_docs", id)
        if rec is None:
            raise NotFound(f"no indexed document {id}")
        for h in rec["hosts"]:
            ctx.world.learn(ctx, ctx.actor, "host", h)
        return {"id": id, "kind": rec["kind"], "title": rec["title"], "text": rec["text"], "owner": rec["owner"],
                "ref": rec["ref"], "hosts": list(rec["hosts"]), "t": rec["t"]}

    # ------------------------------------------------------------------ appliers
    @applier("search.index")
    def ap_index(self, ev):
        p = ev.payload
        st = self.world.state
        old = st.tables["search_docs"].get(p["id"])
        st.put("search_docs", p["id"], {"id": p["id"], "kind": p["kind"], "title": p["title"], "text": p["text"],
                                        "owner": p["owner"], "ref": p["ref"], "hosts": list(p["hosts"]), "t": ev.t,
                                        "seq": ev.seq, "created": old["created"] if old else ev.t})
        self._mark(p["id"])

    @applier("search.unindex")
    def ap_unindex(self, ev):
        self.world.state.delete("search_docs", ev.payload["id"])
        self._mark(ev.payload["id"])

    # ------------------------------------------------------------------ invariants
    def check_invariants(self, world):
        errs = []
        for did, rec in world.state.tables["search_docs"].items():
            if rec.get("id") != did:
                errs.append(f"doc {did} has mismatched id")
            hs = rec.get("hosts")
            if not isinstance(hs, list) or hs != sorted(set(hs)) or len(hs) > MAX_HOSTS:
                errs.append(f"doc {did} hosts not a sorted unique list")
            if len(rec.get("text", "")) > MAX_TEXT or len(rec.get("title", "")) > MAX_TITLE:
                errs.append(f"doc {did} exceeds size limits")
        if not world.state.in_tx:
            if not self._index(world).same_as(_Index(world.state)):
                errs.append("runtime index diverged from a from-scratch rebuild")
        return errs


def _result(did: str, rec, score: float) -> dict:
    return {"id": did, "kind": rec["kind"], "title": rec["title"], "snippet": rec["text"][:SNIPPET],
            "owner": rec["owner"], "ref": rec["ref"], "hosts": list(rec["hosts"]), "t": rec["t"],
            "score": round(score, 4), "sponsored": False}


def _ad_result(ad: dict) -> dict:
    hosts = sorted({str(h) for h in (ad.get("hosts") or [])})[:MAX_HOSTS]
    return {"id": ad.get("id") or f"ad:{ad.get('campaign')}", "campaign": ad.get("campaign"),
            "kind": ad.get("kind") or "ad", "title": str(ad.get("title", ""))[:MAX_TITLE],
            "snippet": str(ad.get("text", ""))[:SNIPPET], "owner": ad.get("owner"), "ref": ad.get("ref"),
            "hosts": hosts, "t": ad.get("t"), "score": None, "sponsored": True}
