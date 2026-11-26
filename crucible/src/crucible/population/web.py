"""Kernel module `popweb`: the population's resource-visit engine (MASTER_SPEC Part 4.1, contract section 2).

Individuals discover web sites (search, links on pages they read, contacts' recommendations, forum posts), visit
them, use them (read pages, evaluate and buy products), return to the ones that were useful, leave the ones that
were not, and move between activities ("web", "search", "forum" - and any activity other modules register).

What a visit is made of - all of it real world state:

* A page is fetched with `net.fetch` from the site's own deployed code, as the first visitor of that page in the tick
  (materialised as principal `p<i>`); the response is recorded (`popweb.page`) and reused by every visitor of the same
  page while it is younger than `page_ttl_s` (default three hours; config) AND the site still runs the same
  deployment in the same availability state (a redeploy, an undeploy, an outage or a suspension invalidates it at
  once). So one recorded fetch serves the cohort that requests the same public page in that window; failures that a
  site's own code produces only at some times are sampled once per page and window.
* What an individual gets from a page is computed from the page itself (its vector in the shared space, recorded
  text encoder, `population.semantics`; its
  length, distinct-word share, links, prices) and the individual's own state (interests, knowledge, preferences,
  skills): relevance, richness, novelty; an error page costs its patience. No site carries a quality score.
* Visits, first visits, returns, abandonment and purchases are recorded decisions: one `cols.batch` on the `pop`
  table (the individuals' new state), one on the `res` table (per-resource counters, `mode="add"`) whose `extra`
  holds the reference of an array blob with one row per visit (who, which resource, via what, utility, pages,
  abandoned, bought) - the visit log the evaluator reads.
* A purchase is the site's own checkout: the individual is materialised with a bank account, its income is
  deposited (`core.money_deposit`), it logs in to the site (`net.login`, its consent), POSTs the buy form (the site
  issues a `bank.invoice` and stores the order in its database) and pays the invoice from its own account
  (`bank.pay_invoice`). No aggregate households account is involved.
* Attention: each page read costs minutes (length and interest), granted through `substrate.attention.spend`
  from the individual's daily budget; when it runs out the session ends.

Class 2 (adaptive) individuals choose which resource to visit with the shared `AdaptivePolicy` of module
`pop_adaptive` (when present); every other decision uses the class 1 models (`models.py`).
"""
from __future__ import annotations

import math
import re

import numpy as np

from ..core import schema as S
from ..core.errors import CrucibleError, NotFound
from ..core.kernel import Ctx, Module, applier, endpoint, process, register_module
from ..substrate import artifacts, attention, columns, ids
from . import semantics, core, schema, text
from .api import record_decisions
from .schema import NONE_I32, RES_TABLE, TABLE

VISIT_MODEL = ("popweb.visit", "1")
PG_TABLE = "webpg"
TOPIC_TABLE = "webtopic"
ARTICLE_OPEN = "article_open_semantic@1"
CUSTOMER_RETURN = "customer_return@1"

DAY = 86400
INVOICE_RE = re.compile(r"(inv_[0-9]+)")


def _host_of_result(r: dict) -> str | None:
    if r.get("kind") == "page" and isinstance(r.get("ref"), str) and ".crucible" in r["ref"]:
        return r["ref"].split(".crucible", 1)[0]
    hs = r.get("hosts") or []
    return hs[0] if hs else None


def kind_of(title: str) -> int:
    """Perceivable category of a site from its title words (sites.CATEGORIES order, 1-based; 0 unknown)."""
    from .sites import CATEGORIES
    toks = set(text.tokens(title))
    for k, c in enumerate(CATEGORIES):
        if c in toks:
            return k + 1
    return 0


def resource_scores(F: dict) -> np.ndarray:
    """Class-1 value of each candidate resource [n, K] plus the value of exploring something new [n, 1] (the
    resource choice of a web visit, `PopWeb._choose`; a Gumbel draw then picks). Inputs per individual and known
    resource: its own satisfaction with it (`known_sat`), how well its content matched the individual's interests
    when last read (`match`, shared space), how often it visited it (`visits`: familiarity with the source),
    `recency` of the last visit, the share of its contacts who like it (`social`); per individual: loyalty, social
    conformity, novelty seeking, a pending suggestion and its source (`suggest`, `suggest_src`), trust in the
    suggesting contact, beliefs about links and unknown sites. Pure function (plausible, unvalidated coefficients;
    FIDELITY_REGISTER pop.decision.resource_choice)."""
    ks, match, kc = F["known_sat"], F["match"], F["visits"]
    valid = F["valid"]
    V = 0.1 + 1.25 * ks + 1.4 * match + 0.25 * F["loyalty"][:, None] * np.log1p(kc) - 0.5 * F["recency"] + \
        0.6 * F["social_conformity"][:, None] * F["social"]
    V = np.where(valid, V, -np.inf)
    best = np.where(valid.any(axis=1), np.max(np.where(valid, ks, -1.0), axis=1), 0.0)
    sug, src = F["suggest"], F["suggest_src"]
    explore = -0.15 + 0.8 * F["novelty_seeking"] + 0.5 * (1.0 - np.clip(best, 0, 1)) + \
        np.where(sug >= 0, 0.3 + 0.6 * np.where(src == 2, F["trust"], F["belief_links"]),
                 0.4 * F["belief_unknown"] - 0.2)
    return np.concatenate([V, explore[:, None]], axis=1)


def _member(a: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Row-wise: is v[i] in a[i, :]? (a [n, k], v [n])."""
    return (a == v[:, None]).any(axis=1) if a.size else np.zeros(v.shape[0], bool)


@register_module
class PopWeb(Module):
    """Resource discovery, visits, use, return, abandonment, purchases; activities `web`, `search`, `forum`.
    Config: {page_ttl_s: 10800, max_depth: 3, max_visits: 5, genesis_known: 2, search_limit: 6, forum: true,
    currency_cents_per_gbp: 100}."""
    name = "popweb"
    host = "popweb"
    version = "1"
    depends = ("pop", "space", "identity", "bank", "code", "cloud", "net", "search", "artifacts", "cols")

    def install(self, world):
        pop = world.modules["pop"]
        self.d = pop.d
        self.D = pop.d["D"]
        columns.declare(world, RES_TABLE, 0, schema.res_columns(pop.d))
        columns.declare(world, TABLE, columns.rows(world, TABLE), schema.web_columns(pop.d))
        st = world.state
        st.declare_table("pop_res", index=("owner",))
        st.declare_table("pop_pages", index=("host",))
        st.declare_table("pop_boards")
        st.declare_table("pop_pagerow")
        # per article page: opens by the population (inputs of article_open_semantic@1); per topic: views on the web
        columns.declare(world, PG_TABLE, 0, {"opens": ("int64", (), 0), "t0": ("int64", (), -1)})
        columns.declare(world, TOPIC_TABLE, len(semantics.topic_names()), {"views": ("int64", (), 0)})
        self.ttl = int(self.config.get("page_ttl_s", 3 * 3600))
        self.max_depth = int(self.config.get("max_depth", 3))
        self.max_visits = int(self.config.get("max_visits", 5))
        # money: world cents per GBP-equivalent unit for customer_return@1 (the contract's declared currency factor)
        self.gbp_cents = float(self.config.get("currency_cents_per_gbp", 100.0))
        core.register_activity(world, "web", self.act_web, owner=self.name, prior=1.0, goals=(1, 2, 4))
        core.register_activity(world, "search", self.act_search, owner=self.name, prior=0.5, goals=(1, 2))
        self.forum_on = bool(self.config.get("forum", True)) and any(m["name"] == "forum" for m in world.spec.modules)
        if self.forum_on:
            core.register_activity(world, "forum", self.act_forum, owner=self.name, prior=0.1, goals=(3, 1))

    def on_load(self, world):
        for k in ("popweb.posts", "popweb.hosts", "popweb.search", "popweb.post_reps", "popweb.active24"):
            world.runtime.pop(k, None)

    # ------------------------------------------------------------------ what people already know at the start
    def genesis(self, ctx):
        # after the sites of the world exist (installed by processes at time zero with a higher priority)
        ctx.schedule(ctx.now, "popweb.seed", {}, prio=20, key="popweb.seed")

    @process("seed")
    def pr_seed(self, ctx):
        st = ctx.state
        hosts = sorted(h for h, rec in st.scan("hosts") if self._is_web(st, h))
        if not hosts:
            return
        rows = self._ensure_res(ctx, hosts)
        k0 = int(self.config.get("genesis_known", 2))
        cls = columns.col(ctx.world, TABLE, "class")
        who = np.flatnonzero(cls <= 2)
        if not who.size or k0 <= 0:
            return
        rng = ctx.world.rng_for("popweb-genesis")
        K = self.d["K"]
        k0 = min(k0, K, rows.size)
        pick = np.argsort(rng.random((who.size, rows.size)), axis=1)[:, :k0]
        known = np.full((who.size, K), NONE_I32, np.int32)
        known[:, :k0] = rows[pick]
        columns.batch(ctx, TABLE, "popweb.genesis", who, {"known": known}, extra={"sites": len(hosts), "per": k0})

    def _is_web(self, st, host) -> bool:
        h = st.get("hosts", host)
        if h is None or not h.get("deployment"):
            return False
        dep = st.get("deployments", h["deployment"])
        return dep is not None and dep["status"] == "live" and "web" in dep["manifest"]

    # ------------------------------------------------------------------ resources
    def _hosts(self, world) -> list[str]:
        """Resource row -> host (runtime list, rebuilt from `pop_res` whenever it changed)."""
        tbl = world.state.tables["pop_res"]
        cache = world.runtime.get("popweb.hosts")
        if cache is None or cache[0] != len(tbl) or cache[2] is not tbl:
            hosts = [""] * len(tbl)
            for h, r in tbl.items():
                hosts[r["row"]] = h
            cache = (len(tbl), hosts, tbl)
            world.runtime["popweb.hosts"] = cache
        return cache[1]

    def _ensure_res(self, ctx: Ctx, hosts: list[str]) -> np.ndarray:
        st = ctx.state
        out = np.full(len(hosts), NONE_I32, np.int64)
        new = []
        for j, h in enumerate(hosts):
            r = st.get("pop_res", h)
            if r is not None:
                out[j] = r["row"]
            elif h and st.get("hosts", h) is not None and h not in new:
                new.append(h)
        if new:
            n0 = columns.rows(ctx.world, RES_TABLE)
            columns.grow(ctx, RES_TABLE, len(new))
            for k, h in enumerate(new):
                ctx.emit("popweb.resource", {"id": h, "row": n0 + k, "owner": st.get("hosts", h)["owner"]},
                         vis="internal")
            pos = {h: n0 + k for k, h in enumerate(new)}
            for j, h in enumerate(hosts):
                if out[j] < 0 and h in pos:
                    out[j] = pos[h]
        return out

    def _outlinks(self, ctx: Ctx, rows: np.ndarray) -> dict[int, np.ndarray]:
        """Resource rows linked from the entry page of each resource in `rows` (as last read)."""
        hosts_all = self._hosts(ctx.world)
        st = ctx.state
        pages = {}
        need = []
        for r in sorted(set(int(x) for x in rows if 0 <= x < len(hosts_all))):
            pg = st.get("pop_pages", f"{hosts_all[r]}|/")
            hs = list(pg["hosts"]) if pg is not None else []
            pages[r] = hs
            need += hs
        rmap = dict(zip(sorted(set(need)), self._ensure_res(ctx, sorted(set(need))))) if need else {}
        return {r: np.array([rmap[h] for h in hs if rmap.get(h, -1) >= 0], np.int64) for r, hs in pages.items()}

    # ------------------------------------------------------------------ pages
    def _dep_state(self, st, host) -> tuple[str | None, str | None]:
        """(live deployment id, availability problem or None) of a host, as the network sees it now."""
        h = st.get("hosts", host)
        dep_id = h.get("deployment") if h is not None else None
        dep = st.get("deployments", dep_id) if dep_id else None
        if dep is None or dep["status"] != "live":
            return None, "down"
        return dep_id, self.world.modules["cloud"].availability(st, dep)

    def _page(self, ctx: Ctx, host: str, path: str, visitor: int) -> dict:
        key = f"{host}|{path}"
        st = ctx.state
        rec = st.get("pop_pages", key)
        if rec is not None and rec["t"] + self.ttl > ctx.now:
            # a cached copy is valid while the site runs the same deployment in the same availability state
            dep_id, why = self._dep_state(st, host)
            if dep_id == rec.get("dep") and why == rec.get("why"):
                return rec
        net = ctx.world.modules["net"]
        p = ids.ensure_principal(ctx, int(visitor))
        try:
            r = ctx.run_as(p, lambda sub: net.fetch(sub, f"http://{host}.crucible{path}", "GET", {}, None, p))
        except CrucibleError as e:
            r = {"status": 599, "content_type": "text/plain", "body": str(e)[:200]}
        status = int(r.get("status", 599))
        body = r.get("body") if isinstance(r.get("body"), str) else ""
        parsed = text.parse_page(body, r.get("content_type", "")) if status == 200 else \
            {"text": "", "title": "", "paths": [], "hosts": [], "prices": []}
        page_text = (parsed["title"] + " " + parsed["text"]).strip()
        tf = semantics.text_features(page_text)
        vec = self._coords(ctx, [page_text])[0] if status == 200 and page_text else np.zeros(self.D, np.float32)
        arts = parsed.get("arts") or []
        # the slate of article headlines a reader sees on the page, represented in the shared space
        av = self._coords(ctx, [a[1] for a in arts]) if (status == 200 and arts) else np.zeros((0, self.D), np.float32)
        dep_id, why = self._dep_state(st, host)
        out = {"key": key, "host": host, "path": path, "status": status, "digest": text.body_digest(body),
               "dep": dep_id, "why": why, "title": parsed["title"], "vec": [round(float(x), 3) for x in vec],
               "chars": len(parsed["text"]), "uniq": round(float(tf["distinct_ratio"]), 3),
               "leg": round(float(semantics.legibility(tf)), 4), "paths": parsed["paths"][:30],
               "hosts": [h for h in parsed["hosts"] if h != host][:12], "prices": parsed["prices"][:10],
               "kind": kind_of(parsed["title"]), "visitor": p,
               "arts": [a[0] for a in arts], "art_vecs": [[round(float(x), 3) for x in v] for v in av],
               "art_topics": [int(t) for t in (semantics.nearest_topic(av) if len(av) else [])]}
        ctx.emit("popweb.page", out, vis="internal")
        return st.get("pop_pages", key)

    def _page_rows(self, ctx: Ctx, keys: list[str]) -> np.ndarray:
        """Rows of article pages (host|path) in the page table, created when a page is first opened."""
        st = ctx.state
        out = np.full(len(keys), -1, np.int64)
        new = []
        for j, k in enumerate(keys):
            r = st.get("pop_pagerow", k)
            if r is not None:
                out[j] = r["row"]
            elif k not in new:
                new.append(k)
        if new:
            n0 = columns.rows(ctx.world, PG_TABLE)
            columns.grow(ctx, PG_TABLE, len(new))
            for q, k in enumerate(new):
                ctx.emit("popweb.pagerow", {"key": k, "row": n0 + q}, vis="internal")
            columns.batch(ctx, PG_TABLE, "popweb.page_seen", np.arange(n0, n0 + len(new)),
                          {"t0": np.full(len(new), ctx.now, np.int64)})
            pos = {k: n0 + q for q, k in enumerate(new)}
            for j, k in enumerate(keys):
                if out[j] < 0:
                    out[j] = pos[k]
        return out

    def _active_24h(self, world) -> int:
        """Individuals who browsed the web in the previous 24 h (a world count, cached per tick)."""
        c = world.runtime.get("popweb.active24")
        if c is not None and c[0] == world.now:
            return c[1]
        v = int(np.count_nonzero(columns.col(world, TABLE, "web_last_at") >= world.now - DAY))
        world.runtime["popweb.active24"] = (world.now, v)
        return v

    def _article_open(self, ctx: Ctx, idx: np.ndarray, inter: np.ndarray, recs: list, which: np.ndarray, rng
                      ) -> tuple[np.ndarray, dict]:
        """The fitted article-open decision (article_open_semantic@1, scope content_kind="article") for readers
        `idx` looking at the article slates of pages `recs[which[j]]`: per reader, the opened article (path index in
        its page's slate, -1 none) and the evaluation record. One evaluation for all (reader, slot) rows; each page's
        slate (headline vectors, topics, page rows, counters) is assembled once, not once per reader."""
        from ..calibration import runtime
        w = ctx.world
        n = idx.size
        pick = np.full(n, -1, np.int64)
        pages = sorted(set(int(x) for x in which.tolist()))
        slate = {}
        for k in pages:
            r = recs[k]
            m = len(r.get("arts") or [])
            if not m:
                continue
            V = np.asarray(r["art_vecs"], np.float32).reshape(m, self.D)
            prow = self._page_rows(ctx, [f"{r['host']}|{a}" for a in r["arts"]])
            slate[k] = (V, np.asarray(r["art_topics"], np.int64), prow)
        sizes = np.array([len(slate[int(k)][1]) if int(k) in slate else 0 for k in which.tolist()], np.int64)
        if not sizes.sum():
            return pick, {}
        rr = np.repeat(np.arange(n), sizes)
        kk = np.concatenate([np.arange(s_) for s_ in sizes if s_ > 0])
        pg = which[rr]
        V = np.concatenate([slate[int(k)][0] for k, s_ in zip(which.tolist(), sizes) if s_ > 0])
        topic = np.concatenate([slate[int(k)][1] for k, s_ in zip(which.tolist(), sizes) if s_ > 0])
        prow = np.concatenate([slate[int(k)][2] for k, s_ in zip(which.tolist(), sizes) if s_ > 0])
        del pg
        vn = np.linalg.norm(V, axis=1)
        cos = np.where(vn > 1e-6, np.einsum("nd,nd->n", inter[rr], V) / np.maximum(vn, 1e-6), 0.0)
        cos = np.clip(cos, -1.0, 1.0)
        # within the reader's OWN slate
        size = np.bincount(rr, minlength=n).astype(np.float64)
        mu = np.bincount(rr, weights=cos, minlength=n) / np.maximum(size, 1)
        sd = np.sqrt(np.bincount(rr, weights=(cos - mu[rr]) ** 2, minlength=n) / np.maximum(size, 1))
        z = np.where(sd[rr] > 1e-9, (cos - mu[rr]) / np.where(sd[rr] > 1e-9, sd[rr], 1.0), 0.0)
        wt = columns.col(w, TABLE, "wtopic")[idx].astype(np.int64)
        wn = columns.col(w, TABLE, "wtopic_n")[idx].astype(np.float64)
        own_t = ((wt[rr] == topic[:, None]) * wn[rr]).sum(axis=1)
        views = columns.col(w, TABLE, "web_views")[idx].astype(np.float64)
        tv = columns.col(w, TOPIC_TABLE, "views").astype(np.float64)
        T = tv.size
        share = (tv[topic] + 1.0) / (tv.sum() + T)
        opens = columns.col(w, PG_TABLE, "opens")[prow].astype(np.float64)
        t0 = columns.col(w, PG_TABLE, "t0")[prow].astype(np.float64)
        age_h = np.maximum(ctx.now - t0, 0.0) / 3600.0
        inputs = {"text_cos": cos, "text_cos_z": z, "own_topic_views": own_t, "own_views_total": views[rr],
                  "topic_population_share": share,
                  "item_opens_24h": opens * np.minimum(1.0, 24.0 / np.maximum(age_h, 1.0)),
                  "active_individuals_24h": np.full(rr.size, float(self._active_24h(w))),
                  "item_age_hours": age_h, "slate_size": size[rr]}
        dm = runtime.get(ARTICLE_OPEN).scoped(content_kind="article")
        res = dm.evaluate(inputs)
        opened = rng.random(rr.size) < res.values
        # the reader goes on with the opened article it found most attractive (the first of several in its slate)
        score = np.where(opened, res.values, -1.0)
        best = np.full(n, -1.0)
        np.maximum.at(best, rr, score)
        hit = np.flatnonzero(opened & (score >= best[rr]) & (best[rr] >= 0))
        first = hit[np.unique(rr[hit], return_index=True)[1]] if hit.size else hit
        pick[rr[first]] = kk[first]
        info = {"rows": rr, "slot": kk, "p": res.values, "opened": opened, "topic": topic, "page_row": prow,
                "provenance": res.provenance()}
        return pick, info

    # ------------------------------------------------------------------ choosing what to visit
    def _choose(self, ctx: Ctx, idx: np.ndarray, now: int, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For each individual: (target resource row or -1, via code, suggester or -1)."""
        w = ctx.world
        c = lambda name: columns.col(w, TABLE, name)[idx]  # noqa: E731
        n = idx.size
        known = c("known").astype(np.int64)
        ks = c("known_sat").astype(np.float64)
        kc = c("known_n").astype(np.float64)
        kl = c("known_last").astype(np.int64)
        prefs = c("prefs").astype(np.float64)
        beliefs = c("beliefs").astype(np.float64)
        inter = c("interests").astype(np.float32)
        valid = known >= 0
        R = columns.rows(w, RES_TABLE)
        safe = np.clip(known, 0, max(R - 1, 0))
        if R:
            rv = columns.col(w, RES_TABLE, "vec")[safe].astype(np.float32)
            match = np.einsum("nd,nkd->nk", inter, rv)          # full-space cosine (interests are unit)
            fetched = columns.col(w, RES_TABLE, "fetched_at")[safe] >= 0
            match = np.where(fetched, match, 0.15)      # never seen: a neutral expectation
        else:
            match = np.zeros(known.shape)
        # social context: share of contacts who know the resource and like it
        contacts = c("contacts").astype(np.int64)
        cvalid = contacts >= 0
        cc = np.clip(contacts, 0, None)
        ck = columns.col(w, TABLE, "known")[cc].astype(np.int64)      # n, C, K
        liked = (columns.col(w, TABLE, "known_sat")[cc] > 0.4) & cvalid[:, :, None]
        ck = np.where(liked, ck, -2)
        social = (ck[:, None, :, :] == known[:, :, None, None]).any(axis=3).sum(axis=2) / \
            np.maximum(cvalid.sum(axis=1), 1)[:, None]
        recency = np.exp(-np.maximum(now - kl, 0) / (4 * 3600.0)) * (kl >= 0)
        sug = c("suggest").astype(np.int64)
        sfrom = c("suggest_from").astype(np.int64)
        src = c("suggest_src").astype(np.int64)
        trust = np.full(n, 0.5)
        if (sfrom >= 0).any():
            slot = contacts == sfrom[:, None]
            tr = c("contact_trust").astype(np.float64)
            trust = np.where(slot.any(axis=1), (tr * slot).sum(axis=1) / np.maximum(slot.sum(axis=1), 1), 0.5)
        scores = resource_scores({"known_sat": ks, "match": match, "visits": kc, "recency": recency, "social": social,
                                  "valid": valid, "loyalty": prefs[:, 5], "social_conformity": prefs[:, 4],
                                  "novelty_seeking": prefs[:, 1], "suggest": sug, "suggest_src": src, "trust": trust,
                                  "belief_links": beliefs[:, 5], "belief_unknown": beliefs[:, 0]})
        V = scores[:, :-1]
        tau = 0.35 + 0.25 * prefs[:, 1:2]
        pick = np.argmax(scores + rng.gumbel(size=scores.shape) * tau, axis=1)
        # class 2 participants choose with their shared adaptive policy
        ad = w.modules.get("pop_adaptive")
        if ad is not None:
            sel = np.flatnonzero(c("class") == 2)
            if sel.size:
                feats = {"inter": inter[sel], "known_sat": ks[sel], "known_n": kc[sel], "match": match[sel],
                         "recency": recency[sel], "social": social[sel], "valid": valid[sel],
                         "known": known[sel], "suggest": sug[sel], "trust": trust[sel], "prefs": prefs[sel],
                         "beliefs": beliefs[sel]}
                pick[sel] = ad.choose(ctx, idx[sel], feats, rng)
        K = known.shape[1]
        target = np.full(n, NONE_I32, np.int64)
        via = np.zeros(n, np.int64)
        frm = np.full(n, NONE_I32, np.int64)
        kk = np.flatnonzero(pick < K)
        target[kk] = known[kk, pick[kk]]
        ex = np.flatnonzero(pick >= K)
        if ex.size:
            aband = c("abandoned")
            # a pending suggestion (a link noticed, a contact's recommendation, a forum post)
            oks = (sug[ex] >= 0) & ~_member(aband[ex], sug[ex]) & ~_member(known[ex], sug[ex])
            e1 = ex[oks]
            target[e1], frm[e1] = sug[e1], sfrom[e1]
            via[e1] = np.select([src[e1] == 1, src[e1] == 3], [1, 4], 2)
            # else an unvisited site linked from a page recently read
            e2 = ex[~oks]
            if e2.size:
                recent = c("recent")[e2, :3]
                links = self._outlinks(ctx, recent.reshape(-1))
                for j, row in zip(e2, recent):
                    cands = np.concatenate([links.get(int(r), np.zeros(0, np.int64)) for r in row if r >= 0] or
                                           [np.zeros(0, np.int64)])
                    cands = cands[~np.isin(cands, known[j]) & ~np.isin(cands, aband[j])]
                    if cands.size:
                        target[j], via[j] = cands[int(rng.integers(0, cands.size))], 1
                    elif valid[j].any():      # nothing new in sight: the best known resource
                        target[j] = known[j, int(np.argmax(V[j]))]
        return target, via, frm

    # ------------------------------------------------------------------ the visit pipeline
    def _visit(self, ctx: Ctx, idx: np.ndarray, rows: np.ndarray, via: np.ndarray, frm: np.ndarray, now: int,
               activity: str, *, budget: np.ndarray | None = None, start: list | None = None,
               intent: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """One visit per individual to resource `rows[j]` (from `start[j]`, default the home page), within the
        session's minutes `budget[j]` and the attention left today. Returns (utility, minutes used)."""
        w = ctx.world
        ds = w.runtime["pop.models"]
        c = lambda name: columns.col(w, TABLE, name)[idx]  # noqa: E731
        n = idx.size
        D = self.D
        rng = ctx.rng("popweb.visit", activity)
        hosts_all = self._hosts(w)
        inter = c("interests").astype(np.float32)
        know = c("knowledge").astype(np.float32)
        prefs = c("prefs").astype(np.float64)
        skills = c("skills").astype(np.float64)
        beliefs = c("beliefs").astype(np.float64)
        buyer = (c("goals") == 2).any(axis=1).astype(np.float64)
        wtp_s = c("wtp_scale").astype(np.float64)
        income = c("income_cents").astype(np.float64)
        bud = c("attention_budget_min").astype(np.float64)
        left = c("attention_left_min").astype(np.float64)
        if budget is not None:
            left = np.minimum(left, np.asarray(budget, np.float64))
        intent = np.zeros(n, bool) if intent is None else np.asarray(intent, bool)
        path = list(start) if start is not None else ["/"] * n
        active = rows >= 0
        u_sum = np.zeros(n)
        pages = np.zeros(n, np.int64)
        okp = np.zeros(n, np.int64)
        minutes = np.zeros(n)
        consumed = np.zeros((n, D), np.float32)
        bought = np.zeros(n, bool)
        paid = np.zeros(n, np.int64)
        prod_of = np.full(n, -1, np.int64)
        evaluated = np.zeros(n, bool)
        match0 = np.zeros(n, np.float32)     # interest match with the entry page, at the time of the decision
        new_seen = {}            # row -> latest entry-page record (refreshes the res table)
        reads = {"rows": [], "topic": [], "page_row": []}       # articles opened through the fitted slate decision
        slate_rows = np.zeros(n, bool)
        art_prov = None
        for depth in range(self.max_depth + 1):
            act = np.flatnonzero(active & (left > 0.2))
            if not act.size:
                break
            keys = np.array([f"{hosts_all[rows[j]]}|{path[j]}" for j in act])
            ukeys, first_i, inv = np.unique(keys, return_index=True, return_inverse=True)
            recs = [self._page(ctx, *k.split("|", 1), int(idx[act[f]])) for k, f in zip(ukeys, first_i)]
            P_ok = np.array([r["status"] == 200 for r in recs])
            P_V = np.array([r["vec"] for r in recs], np.float32).reshape(len(recs), D)
            P_chars = np.array([r["chars"] for r in recs], np.float64)
            P_uniq = np.array([r["uniq"] for r in recs], np.float64)
            P_leg = np.array([float(r.get("leg", 0.5)) for r in recs], np.float64)
            P_prods = [[p for p in r["paths"] if p.startswith("/p/")] for r in recs]
            P_arts = [[p for p in r["paths"] if p.startswith("/a/") or p == "/shop"] for r in recs]
            P_slate = np.array([bool(r.get("arts")) and r["status"] == 200 for r in recs])
            P_price = np.array([(r["prices"][0] if r["prices"] and r["path"].startswith("/p/") else -1) for r in recs])
            for k, r in enumerate(recs):
                if r["path"] == "/":
                    new_seen[int(rows[act[first_i[k]]])] = r
            ok, V, chars, uniq, leg = P_ok[inv], P_V[inv], P_chars[inv], P_uniq[inv], P_leg[inv]
            rel = np.einsum("nd,nd->n", inter[act], V.astype(np.float32)) * ok
            if depth == 0:
                match0[act] = rel
            # what a reader gets from a page: relevance to its interests (shared space) x how much readable content
            # it holds (length and the text's intrinsic legibility: real words, natural function-word use, little
            # repetition - keyword lists and word salads score low), plus novelty for those who seek it
            rich = np.sqrt(np.minimum(1.0, chars / 2000.0)) * leg
            nov = 1.0 - np.clip((know[act] * np.abs(V)).sum(axis=1) / np.maximum(np.abs(V).sum(axis=1), 1e-6), 0, 1)
            qs, tol, nseek = prefs[act, 3], prefs[act, 2], prefs[act, 1]
            u_ok = (0.1 + 1.5 * rel) * (0.3 + 0.7 * rich * (0.5 + qs)) + 0.15 * nov * nseek - 0.05 \
                - 0.3 * skills[act, 2] * (uniq < 0.3)
            u = np.where(ok, u_ok, -(0.6 - 0.4 * tol)) + rng.normal(0, 0.12, act.size)
            dwell = np.minimum(np.where(ok, np.clip(chars / 900.0 * (0.5 + np.maximum(rel, 0)), 0.3, 8.0), 0.3),
                               left[act])
            left[act] -= dwell
            minutes[act] += dwell
            u_sum[act] += u
            pages[act] += 1
            okp[act] += ok
            consumed[act] += V * np.maximum(u, 0)[:, None].astype(np.float32)
            # a visitor on a product page decides whether to buy it (through the site's checkout); a customer who came
            # back to buy again (customer_return@1 intention) buys what it intended to if it can
            price = P_price[inv]
            on_prod = np.flatnonzero(ok & (price > 0) & ~bought[act])
            if on_prod.size:
                j = act[on_prod]
                evaluated[j] = True
                pr = price[on_prod].astype(np.float64)
                wallet = core.spendable(w, idx[j]).astype(np.float64)
                # willingness to pay: a share of the individual's own daily income, scaled by how well the product's
                # own description matches its interests
                wtp = np.maximum(0.4 * income[j] * wtp_s[j] * (1.0 + 1.5 * np.maximum(rel[on_prod], 0.0)), 50.0)
                F = {"price_ratio": np.clip((wtp - pr) / wtp, -3, 1), "page_relevance": rel[on_prod],
                     "goal_buy": buyer[j], "wallet_cover_log": np.clip(np.log(np.maximum(wallet, 1) / pr), -5, 5),
                     "price_sensitivity": prefs[j, 0], "resource_satisfaction": self._sat_of(c, j, rows[j]),
                     "belief_shops": beliefs[j, 3], "impulsiveness": prefs[j, 7]}
                buy = (ds["purchase"].sample(F, rng) | intent[j]) & (wallet >= pr)
                for jj, a, pp in zip(j[buy], on_prod[buy], pr[buy]):
                    got = self._purchase(ctx, int(idx[jj]), hosts_all[rows[jj]], path[jj], int(pp))
                    if got:
                        bought[jj] = True
                        paid[jj] = got
                        try:
                            prod_of[jj] = int(path[jj].split("/")[2])
                        except (IndexError, ValueError):
                            prod_of[jj] = -1
                        wa = wtp[a == on_prod][0]
                        u_sum[jj] += 0.4 + 0.4 * float(np.clip((wa - pp) / wa, -1, 1))
            # continue reading? On a page with an article slate the fitted article-open decision says which headline
            # (if any) the reader opens; elsewhere (an article, a shop) the plausible continue/evaluate decisions
            nxt_path = [None] * act.size
            sl = np.flatnonzero(P_slate[inv] & ok & (depth < self.max_depth) & (left[act] > 0.2))
            if sl.size:
                slate_rows[act[sl]] = True
                pk, info = self._article_open(ctx, idx[act[sl]], inter[act[sl]], recs, inv[sl], rng)
                art_prov = info.get("provenance") or art_prov
                for q, (a, k_) in enumerate(zip(sl, pk)):
                    if k_ >= 0:
                        nxt_path[a] = recs[inv[a]]["arts"][int(k_)]
                if info:
                    # the article each reader goes on with (its pick) counts as a view of its topic and page
                    sel_ = np.flatnonzero(pk[info["rows"]] == info["slot"])
                    reads["rows"] += act[sl[info["rows"][sel_]]].tolist()
                    reads["topic"] += info["topic"][sel_].tolist()
                    reads["page_row"] += info["page_row"][sel_].tolist()
            Fc = {"page_utility": u, "attention_ratio": left[act] / np.maximum(bud[act], 1e-3),
                  "quality_sensitivity": qs, "novelty_seeking": nseek, "page_richness": rich}
            cont = ds["continue"].sample(Fc, rng) & ok & (depth < self.max_depth) & ~P_slate[inv]
            Fe = {"page_relevance": rel, "goal_buy": np.maximum(buyer[act], intent[act]), "impulsiveness": prefs[act, 7],
                  "shopping_skill": skills[act, 3], "attention_ratio": left[act] / np.maximum(bud[act], 1e-3)}
            want = ds["evaluate_product"].sample(Fe, rng) | intent[act]
            r_pick = rng.random(act.size)
            active[:] = False
            for a in range(act.size):
                j = act[a]
                nxt = nxt_path[a]
                if nxt is None and cont[a]:
                    k = inv[a]
                    lst = P_prods[k] if (want[a] and P_prods[k]) else P_arts[k]
                    if lst:
                        nxt = lst[int(r_pick[a] * len(lst))]
                if nxt is not None and nxt != path[j]:
                    path[j] = nxt
                    active[j] = True
        U = np.where(pages > 0, u_sum / np.maximum(pages, 1), -0.1)
        self._record(ctx, idx, rows, via, frm, U, pages, okp, minutes, consumed, bought, paid, evaluated, new_seen,
                     now, activity, rng, match0, prod_of=prod_of, intent=intent)
        if art_prov is not None:
            self._record_reads(ctx, idx, slate_rows, reads, art_prov, now)
        items = w.runtime.get("pop.tick.items")
        if items is not None:
            for i, p_ in zip(idx[pages > 0].tolist(), pages[pages > 0].tolist()):
                items[int(i)] = items.get(int(i), 0) + int(p_)
        ad = w.modules.get("pop_adaptive")
        if ad is not None:
            ad.reward(ctx, idx, U)          # adaptive participants learn from the utility they experienced
        return U, minutes

    def _record_reads(self, ctx: Ctx, idx: np.ndarray, slate_rows: np.ndarray, reads: dict, prov: dict, now: int):
        """The article-open decisions of a visit (article_open_semantic@1): the readers' reading history (views, the
        topics they read), the pages' and topics' open counts."""
        w = ctx.world
        who = np.flatnonzero(slate_rows)
        if not who.size:
            return
        views = columns.col(w, TABLE, "web_views")[idx[who]].astype(np.int64)
        wt = columns.col(w, TABLE, "wtopic")[idx[who]].astype(np.int64).copy()
        wn = columns.col(w, TABLE, "wtopic_n")[idx[who]].astype(np.int64).copy()
        pos = {int(j): q for q, j in enumerate(who)}
        for j, t in zip(reads["rows"], reads["topic"]):
            q = pos[int(j)]
            views[q] += 1
            hit = np.flatnonzero(wt[q] == t)
            if hit.size:
                wn[q, hit[0]] += 1
            else:                                  # the least read of the four tracked topics makes room
                k = int(np.argmin(np.where(wt[q] >= 0, wn[q], -1)))
                wt[q, k], wn[q, k] = t, 1
        from .activation import provenance_args
        record_decisions(ctx, "popweb.article_open", idx[who], {
            "web_views": np.minimum(views, 2**31 - 1).astype(np.int32), "wtopic": wt.astype(np.int16),
            "wtopic_n": np.minimum(wn, 65535).astype(np.uint16), "web_last_at": np.full(who.size, now, np.int64)},
            **provenance_args(prov), extra={"t": int(now), "readers": int(who.size), "opened": len(reads["rows"]),
                                            "served": prov.get("served"), "extrapolated": int(prov.get("extrapolated", 0))})
        if reads["page_row"]:
            pr, cnt = np.unique(np.array(reads["page_row"], np.int64), return_counts=True)
            columns.batch(ctx, PG_TABLE, "popweb.page_opens", pr, {"opens": cnt.astype(np.int64)}, mode="add",
                          extra={"t": int(now)})
            tp, tc = np.unique(np.array(reads["topic"], np.int64), return_counts=True)
            columns.batch(ctx, TOPIC_TABLE, "popweb.topic_views", tp, {"views": tc.astype(np.int64)}, mode="add",
                          extra={"t": int(now)})

    def _sat_of(self, c, j: np.ndarray, rows: np.ndarray) -> np.ndarray:
        kn = c("known")[j]
        hit = kn == rows[:, None]
        return np.where(hit.any(axis=1), (c("known_sat")[j].astype(np.float64) * hit).sum(axis=1), 0.0)

    # ------------------------------------------------------------------ purchases (bank, the site's checkout)
    def _purchase(self, ctx: Ctx, i: int, host: str, path: str, price: int) -> int:
        """Buy through the site: deposit income, log in, POST the buy form, pay the invoice. Returns cents paid."""
        w = ctx.world
        net = w.modules["net"]
        bank = w.modules["bank"]
        try:
            acct = core.money_deposit(ctx, i)
            p = ids.pid(i)
            if acct is None or bank.balance(ctx.state, acct) < price:
                return 0
            sess = ctx.state.get("web_sessions", f"{host}|{p}")
            if sess is None or not sess.get("login"):
                ctx.call("net.login", {"site": host}, as_principal=p)
            r = ctx.run_as(p, lambda sub: net.fetch(sub, f"http://{host}.crucible{path}/buy", "POST", {}, None, p))
            m = INVOICE_RE.search(r.get("body") or "") if r.get("status") == 200 else None
            if m is None:
                return 0
            inv = ctx.state.get("invoices", m.group(1))
            if inv is None or inv["payer"] != p or inv["status"] != "open":
                return 0
            ctx.call("bank.pay_invoice", {"invoice": m.group(1), "from_account": acct}, as_principal=p)
            return int(inv["amount"])
        except CrucibleError:
            return 0

    # ------------------------------------------------------------------ recording what happened
    def _record(self, ctx, idx, rows, via, frm, U, pages, okp, minutes, consumed, bought, paid, evaluated, new_seen,
                now, activity, rng, match0, *, prod_of=None, intent=None):
        w = ctx.world
        ds = w.runtime["pop.models"]
        c = lambda name: columns.col(w, TABLE, name)[idx]  # noqa: E731
        n = idx.size
        has = (rows >= 0) & (pages > 0)
        known0, ks0 = c("known"), c("known_sat")
        known = known0.copy()
        ks = ks0.astype(np.float32)
        kn = c("known_n").copy()
        kl = c("known_last").copy()
        prefs = c("prefs").astype(np.float64)
        hit = known == rows[:, None]
        is_known = hit.any(axis=1) & has
        first = has & ~is_known
        slot = np.where(is_known, np.argmax(hit, axis=1), 0)
        if first.any():
            empty = known < 0
            worst = np.argmin(np.where(empty, -9.0, ks), axis=1)
            slot = np.where(first, np.where(empty.any(axis=1), np.argmax(empty, axis=1), worst), slot)
        r_ = np.flatnonzero(has)
        sl = slot[r_]
        alpha = 0.35 + 0.3 * (1 - prefs[r_, 5])
        new = np.where(first[r_], U[r_], (1 - alpha) * ks[r_, sl].astype(np.float64) + alpha * U[r_])
        known[r_, sl] = rows[r_]
        ks[r_, sl] = new
        kn[r_, sl] = np.where(first[r_], 1, np.minimum(kn[r_, sl].astype(np.int64) + 1, 65000))
        kl[r_, sl] = now
        # abandonment: leave a resource that falls short of what the individual has come to expect
        sat = c("satisfaction").astype(np.float64)
        tol = np.clip(0.6 * sat[r_] + 0.15 - 0.25 * prefs[r_, 2], 0.05, 0.6)
        gap = tol - new
        Fa = {"tolerance_gap": gap, "loyalty": prefs[r_, 5], "resource_visits_log": np.log1p(kn[r_, sl].astype(float))}
        ab = ds["abandon"].sample(Fa, rng) & (gap > -0.05)
        aband = c("abandoned").copy()
        ab_r = r_[ab]
        if ab_r.size:
            aband[ab_r, 1:] = aband[ab_r, :-1]
            aband[ab_r, 0] = rows[ab_r]
            known[ab_r, slot[ab_r]] = NONE_I32
            ks[ab_r, slot[ab_r]] = 0.0
            kn[ab_r, slot[ab_r]] = 0
            kl[ab_r, slot[ab_r]] = -1
        abandoned = np.zeros(n, bool)
        abandoned[ab_r] = True
        recent = c("recent").copy()
        recent[r_, 1:] = recent[r_, :-1]
        recent[r_, 0] = rows[r_]
        # suggestions taken: trust in the recommender and beliefs about the channel follow the experience
        sug0 = c("suggest")
        sug = sug0.astype(np.int64)
        sfrom = c("suggest_from").astype(np.int64)
        ssrc = c("suggest_src").astype(np.int64)
        used = has & (via > 0) & (rows == sug)
        trust0 = c("contact_trust")
        trust = trust0.astype(np.float32)
        contacts = c("contacts")
        beliefs0 = c("beliefs")
        beliefs = beliefs0.astype(np.float32)
        good = (U > 0.3).astype(np.float32)
        uc = np.flatnonzero(used & (via == 2) & (sfrom >= 0))
        if uc.size:
            m = contacts[uc] == sfrom[uc, None]
            trust[uc] = np.where(m, np.clip(trust[uc] + (0.15 * (U[uc] - 0.3))[:, None], 0.02, 0.98), trust[uc])
            beliefs[uc, 2] = 0.85 * beliefs[uc, 2] + 0.15 * good[uc]
        ul = np.flatnonzero(used & (via == 1))
        beliefs[ul, 5] = 0.85 * beliefs[ul, 5] + 0.15 * good[ul]
        uf = np.flatnonzero(used & (via == 4))
        beliefs[uf, 4] = 0.85 * beliefs[uf, 4] + 0.15 * good[uf]
        sug = np.where(used, NONE_I32, sug)
        sfrom = np.where(used, NONE_I32, sfrom)
        ssrc = np.where(used, 0, ssrc)
        beliefs[first, 0] = 0.9 * beliefs[first, 0] + 0.1 * good[first]
        # discovery: an external site linked from the pages read (noticed with the individual's novelty seeking)
        notice = np.flatnonzero(has & (rng.random(n) < 0.25 + 0.5 * prefs[:, 1]) & (sug < 0))
        if notice.size:
            links = self._outlinks(ctx, rows[notice])
            pick = rng.random(notice.size)
            for j, pk in zip(notice, pick):
                ls = links.get(int(rows[j]))
                if ls is not None and ls.size:
                    r = ls[int(pk * ls.size)]
                    if r not in known[j] and r not in aband[j]:
                        sug[j], sfrom[j], ssrc[j] = r, NONE_I32, 1
        # preferences evolve: interests move towards content that was enjoyed, knowledge accumulates
        inter = c("interests").astype(np.float32)
        know = c("knowledge").astype(np.float32)
        nrm = np.linalg.norm(consumed, axis=1, keepdims=True)
        liked = (nrm[:, 0] > 1e-6) & has
        eta = (0.02 + 0.06 * prefs[:, 1:2]).astype(np.float32)
        direc = np.where(liked[:, None], consumed / np.maximum(nrm, 1e-6), 0).astype(np.float32)
        inter = np.where(liked[:, None], inter + eta * (direc - inter), inter)
        inter = inter / np.maximum(np.linalg.norm(inter, axis=1, keepdims=True), 1e-6)
        know = np.clip(know + (0.03 * np.minimum(minutes, 10.0) / 10.0)[:, None].astype(np.float32) * np.abs(direc), 0, 1)
        skills0 = c("skills")
        skills = skills0.astype(np.float32)
        disl = has & (U < -0.1) & (okp > 0)
        skills[has, 6] = np.clip(skills[has, 6] + 0.002 * pages[has], 0, 1)              # reading
        skills[disl, 2] = np.clip(skills[disl, 2] + 0.01, 0, 1)                          # evaluating
        skills[bought, 3] = np.clip(skills[bought, 3] + 0.02, 0, 1)                      # shopping
        beliefs[bought, 3] = 0.9 * beliefs[bought, 3] + 0.1
        # preferences adapt: exploring that paid off makes novelty more attractive, failures make one less patient,
        # good returns build loyalty, purchases make one more price-aware
        pf0 = c("prefs")
        pf = pf0.astype(np.float32)
        dev = (U - sat).astype(np.float32)
        pf[first, 1] += 0.02 * dev[first]
        pf[~first & has, 5] += 0.01 * dev[~first & has]
        fail = has & (okp < pages)
        pf[fail, 2] -= 0.01
        pf[bought, 0] += 0.01
        pf = np.clip(pf, 0.02, 0.98)
        vals = {"known": known, "known_sat": ks.astype(np.float16), "known_n": kn, "known_last": kl,
                "prefs": pf.astype(np.float16),
                "abandoned": aband, "recent": recent, "suggest": sug.astype(np.int32),
                "suggest_from": sfrom.astype(np.int32), "suggest_src": ssrc.astype(np.uint8),
                "contact_trust": trust.astype(np.float16), "beliefs": beliefs.astype(np.float16),
                "interests": inter.astype(np.float16), "knowledge": know.astype(np.float16),
                "skills": skills.astype(np.float16)}
        if bought.any():
            b_ix = np.flatnonzero(bought)
            goals = c("goals").copy()
            goals[b_ix, 0] = np.where(goals[b_ix, 0] == 2, 4, goals[b_ix, 0])        # a buying goal is met
            vals.update({"goals": goals, "n_purchases": (c("n_purchases") + bought).astype(np.int32),
                         "spent_cents": (c("spent_cents") + paid).astype(np.int64)})
        intent = np.zeros(n, bool) if intent is None else np.asarray(intent, bool)
        if intent.any():                       # a repurchase intention is used up by the visit it caused
            vals.update({"intent_res": np.where(intent, NONE_I32, c("intent_res")).astype(np.int32),
                         "intent_at": np.where(intent, core.NEVER, c("intent_at")).astype(np.int64),
                         "intent_prod": np.where(intent, -1, c("intent_prod")).astype(np.int16)})
        vals["web_last_at"] = np.where(has, now, c("web_last_at")).astype(np.int64)
        for k in ("suggest_from", "suggest_src", "contact_trust", "abandoned", "prefs"):
            if np.array_equal(vals[k], c(k)):
                del vals[k]            # unchanged columns are not rewritten
        if minutes.sum() > 0:
            mi = np.flatnonzero(minutes > 0)
            attention.spend(ctx, idx[mi], minutes[mi], activity=activity)
        # the visit log (one row per visit) and per-resource counters
        v = np.flatnonzero(has)
        cls = c("class")
        log = {"ind": idx[v].astype(np.int32), "res": rows[v].astype(np.int32), "via": via[v].astype(np.uint8),
               "first": first[v], "util": U[v].astype(np.float32), "pages": pages[v].astype(np.int16),
               "ok_pages": okp[v].astype(np.int16), "abandon": abandoned[v], "bought": bought[v],
               "paid": paid[v], "evaluated": evaluated[v], "cls": cls[v], "minutes": minutes[v].astype(np.float32),
               "match": match0[v]}
        ref = artifacts.put_array_blob(ctx, log) if v.size else None
        record_decisions(ctx, f"popweb.{activity}", idx, vals, mode="set", model_id=VISIT_MODEL[0],
                         model_version=VISIT_MODEL[1], fidelity_id="pop.decision.visit_outcome",
                         extra={"t": int(now), "visits": int(v.size), "first": int(first.sum()),
                                "abandoned": int(abandoned.sum()), "bought": int(bought.sum()), "visits_ref": ref})
        if bought.any():
            self._repurchase(ctx, idx, rows, bought, paid, prod_of, now, rng)
        if v.size:
            u_rows, inv = np.unique(rows[v], return_inverse=True)
            agg = {k: np.zeros(u_rows.size, np.int64) for k in
                   ("visits", "first_visits", "return_visits", "abandons", "purchases", "revenue_cents", "ok_n",
                    "err_n")}
            np.add.at(agg["visits"], inv, 1)
            np.add.at(agg["first_visits"], inv, first[v].astype(np.int64))
            np.add.at(agg["return_visits"], inv, (~first[v]).astype(np.int64))
            np.add.at(agg["abandons"], inv, abandoned[v].astype(np.int64))
            np.add.at(agg["purchases"], inv, bought[v].astype(np.int64))
            np.add.at(agg["revenue_cents"], inv, paid[v])
            np.add.at(agg["ok_n"], inv, okp[v])
            np.add.at(agg["err_n"], inv, pages[v] - okp[v])
            columns.batch(ctx, RES_TABLE, f"popweb.visits.{activity}", u_rows, agg, mode="add",
                          extra={"t": int(now), "visits_ref": ref, "day": int(now // DAY)})
        if new_seen:
            rs = np.array(sorted(new_seen), np.int64)
            recs = [new_seen[int(r)] for r in rs]
            columns.batch(ctx, RES_TABLE, "popweb.seen", rs, {
                "vec": np.array([r["vec"] for r in recs], np.float32).astype(np.float16),
                "chars": np.array([r["chars"] for r in recs], np.int32),
                "uniq": np.array([r["uniq"] for r in recs], np.float16),
                "links": np.array([len(r["hosts"]) for r in recs], np.int16),
                "items": np.array([sum(1 for p in r["paths"] if p.startswith("/p/") or p == "/shop") for r in recs],
                                  np.int16),
                "kind": np.array([r["kind"] for r in recs], np.uint8),
                "status": np.array([r["status"] for r in recs], np.int16),
                "fetched_at": np.array([r["t"] for r in recs], np.int64)}, mode="set")
        # word of mouth: satisfied visitors recommend the resource to a contact
        Fr = {"page_utility": U, "sociability": prefs[:, 6], "resource_satisfaction": ks[np.arange(n), slot]}
        rj = np.flatnonzero(ds["recommend"].sample(Fr, rng) & has & ~abandoned)
        if rj.size:
            cc = contacts[rj]
            pick = cc[np.arange(rj.size), rng.integers(0, cc.shape[1], rj.size)].astype(np.int64)
            ok = pick >= 0
            to, fr, rs_ = pick[ok], idx[rj[ok]], rows[rj[ok]]
            if to.size:
                free = columns.col(w, TABLE, "suggest")[to] < 0
                to, fr, rs_ = to[free], fr[free], rs_[free]
                if to.size:
                    o = np.argsort(to, kind="stable")
                    # the recommenders decided (they are in the session cohort); the recipients' suggestion slots
                    # are where the recommendation lands (delivery)
                    record_decisions(ctx, "popweb.recommend", to[o], {
                        "suggest": rs_[o].astype(np.int32), "suggest_from": fr[o].astype(np.int32),
                        "suggest_src": np.full(to.size, 2, np.uint8)}, model_id="pop.recommend", model_version="1",
                        fidelity_id="pop.decision.recommend", extra={"t": int(now), "n": int(to.size),
                                                                     "by": [int(x) for x in fr[o][:64]]})

    # ------------------------------------------------------------------ repurchase (customer_return@1)
    def _repurchase(self, ctx: Ctx, idx, rows, bought, paid, prod_of, now: int, rng) -> None:
        """Customers who bought today: their order history with the shop (`purchased` slots and `pur_*`) and the
        fitted decision whether they order from the same shop again within 90 days (customer_return@1, UCI Online
        Retail II). A customer who will return gets a repurchase intention (the shop, the product, a day within 90
        drawn from its own mean gap between orders, or 30 days for a first order) and a wake-up then; the return
        visit and the purchase go through the site's own pages and checkout, so a shop that stopped working loses
        the order."""
        from ..calibration import runtime
        from .activation import month, provenance_args
        w = ctx.world
        b = np.flatnonzero(bought)
        ii = idx[b]
        today = int(now // DAY)
        P = columns.col(w, TABLE, "purchased")[ii].astype(np.int64).copy()
        cols_ = {k: columns.col(w, TABLE, k)[ii].astype(np.int64).copy()
                 for k in ("pur_n", "pur_first", "pur_last", "pur_spend", "pur_gapsum", "pur_prod")}
        slot = np.zeros(b.size, np.int64)
        for q in range(b.size):
            hit = np.flatnonzero(P[q] == rows[b[q]])
            if hit.size:
                slot[q] = hit[0]
            else:                          # a new shop: the oldest slot makes room
                P[q, 1:] = P[q, :-1]
                for k in cols_:
                    cols_[k][q, 1:] = cols_[k][q, :-1]
                P[q, 0] = rows[b[q]]
                cols_["pur_n"][q, 0] = 0
                cols_["pur_first"][q, 0] = -1
                cols_["pur_last"][q, 0] = -1
                cols_["pur_spend"][q, 0] = 0
                cols_["pur_gapsum"][q, 0] = 0
                slot[q] = 0
        r = np.arange(b.size)
        n_before = cols_["pur_n"][r, slot].astype(np.float64)
        last = cols_["pur_last"][r, slot]
        first = cols_["pur_first"][r, slot]
        same_day = (last == today) & (n_before > 0)
        orders_before = np.where(same_day, n_before - 1, n_before)
        gaps = cols_["pur_gapsum"][r, slot].astype(np.float64) + np.where((last >= 0) & ~same_day, today - last, 0)
        spend_today_c = paid[b].astype(np.float64)
        spend_total_c = cols_["pur_spend"][r, slot].astype(np.float64) + spend_today_c
        inputs = {"orders_before": orders_before,
                  "tenure_days": np.where(first >= 0, today - first, 0).astype(np.float64),
                  "recency_days": np.where((last >= 0) & ~same_day, today - last, 0).astype(np.float64),
                  "mean_gap_days": np.where(orders_before >= 2, gaps / np.maximum(orders_before, 1), 0.0),
                  "spend_today": spend_today_c / self.gbp_cents, "spend_total": spend_total_c / self.gbp_cents,
                  "lines_today": np.ones(b.size), "products_today": np.ones(b.size), "cancel_share": np.zeros(b.size),
                  "home_market": np.ones(b.size), "month": np.full(b.size, float(month(today)))}
        dm = runtime.get(CUSTOMER_RETURN)
        res = dm.evaluate(inputs)
        again = rng.random(b.size) < res.values
        # update the order history (one order day per day)
        cols_["pur_n"][r, slot] = np.where(same_day, n_before, n_before + 1).astype(np.int64)
        cols_["pur_first"][r, slot] = np.where(first >= 0, first, today)
        cols_["pur_gapsum"][r, slot] = gaps.astype(np.int64)
        cols_["pur_last"][r, slot] = today
        cols_["pur_spend"][r, slot] = spend_total_c.astype(np.int64)
        cols_["pur_prod"][r, slot] = np.asarray(prod_of, np.int64)[b]
        mean_gap = np.where(orders_before >= 1, np.maximum(gaps / np.maximum(orders_before, 1), 1.0), 30.0)
        day_ahead = np.clip(np.ceil(rng.exponential(mean_gap)), 1, 90).astype(np.int64)
        at = now + day_ahead * DAY + rng.integers(0, 6 * 3600, b.size)
        cur_at = columns.col(w, TABLE, "intent_at")[ii].astype(np.int64)
        cur_res = columns.col(w, TABLE, "intent_res")[ii].astype(np.int64)
        cur_prod = columns.col(w, TABLE, "intent_prod")[ii].astype(np.int64)
        take = again & (at < cur_at)              # the earliest intention is kept
        vals = {"purchased": P.astype(np.int32), "pur_n": np.minimum(cols_["pur_n"], 65535).astype(np.uint16),
                "pur_first": cols_["pur_first"].astype(np.int32), "pur_last": cols_["pur_last"].astype(np.int32),
                "pur_spend": cols_["pur_spend"].astype(np.int64), "pur_gapsum": cols_["pur_gapsum"].astype(np.int32),
                "pur_prod": cols_["pur_prod"].astype(np.int16),
                "intent_res": np.where(take, rows[b], cur_res).astype(np.int32),
                "intent_at": np.where(take, at, cur_at).astype(np.int64),
                "intent_prod": np.where(take, np.asarray(prod_of, np.int64)[b], cur_prod).astype(np.int16)}
        pv = res.provenance()
        record_decisions(ctx, "popweb.repurchase", ii, vals, **provenance_args(pv),
                         extra={"t": int(now), "customers": int(b.size), "will_return": int(again.sum()),
                                "served": pv.get("served"), "extrapolated": int(pv.get("extrapolated", 0))})
        if take.any():
            core.wake(ctx, ii[take], at[take], "web")

    # ------------------------------------------------------------------ activities
    def act_web(self, ctx: Ctx, idx: np.ndarray, now: int) -> np.ndarray:
        """A web session: up to `max_visits` visits while the session's minutes last (`pop.session_budget_min`, the
        population's session-length decision) and attention is left; a customer whose repurchase intention is due
        goes to that shop's product first."""
        w = ctx.world
        rng = ctx.rng("popweb.choose")
        n = idx.size
        budget = core.session_budget(w, idx).astype(np.float64)
        U_sum = np.zeros(n)
        n_vis = np.zeros(n)
        live = np.ones(n, bool)
        for k in range(max(1, self.max_visits)):
            sel = np.flatnonzero(live & (budget > (0.0 if k == 0 else 1.0)))
            if not sel.size:
                break
            ii = idx[sel]
            rows, via, frm = self._choose(ctx, ii, ctx.now, rng)
            start, intent = None, None
            if k == 0:
                ir = columns.col(w, TABLE, "intent_res")[ii].astype(np.int64)
                ia = columns.col(w, TABLE, "intent_at")[ii].astype(np.int64)
                ip = columns.col(w, TABLE, "intent_prod")[ii].astype(np.int64)
                intent = (ir >= 0) & (ia <= ctx.now)
                if intent.any():
                    rows = np.where(intent, ir, rows)
                    via = np.where(intent, 0, via)
                    frm = np.where(intent, NONE_I32, frm)
                    start = [f"/p/{int(p)}" if (it and p >= 0) else "/" for it, p in zip(intent, ip)]
            U, mins = self._visit(ctx, ii, rows, via, frm, now, "web", budget=budget[sel], start=start, intent=intent)
            budget[sel] = budget[sel] - mins
            got = rows >= 0
            U_sum[sel] += np.where(got, U, 0.0)
            n_vis[sel] += got
            # a frustrating visit (or nothing to visit) ends the session
            live[sel] = got & (U > -0.2) & (mins > 0)
        return np.where(n_vis > 0, U_sum / np.maximum(n_vis, 1), -0.1)

    def _search(self, ctx: Ctx, q: str, lim: int, version) -> list[tuple[str, np.ndarray]]:
        """Organic search results for `q` among indexed web pages: [(host, snippet vector)]. Cached per version of
        the page index (number of indexed pages and the last page-indexing event; pages change when the crawler runs),
        so a cache hit equals a fresh search."""
        world = ctx.world
        cache = world.runtime.setdefault("popweb.search", {})
        if cache.get("version") != version:
            cache.clear()
            cache["version"] = version
        hit = cache.get(q)
        if hit is not None:
            return hit
        _, organic, _ = world.modules["search"].search(world, q, "page", lim)
        hs, seen, snippets = [], set(), []
        for r in organic:
            h = _host_of_result(r)
            if h and h not in seen and self._is_web(world.state, h):
                seen.add(h)
                hs.append(h)
                snippets.append((r.get("snippet", "") + " " + r.get("title", "")).strip() or h)
        vecs = self._coords(ctx, snippets) if snippets else np.zeros((0, self.D), np.float32)
        hs = [(h, vecs[j]) for j, h in enumerate(hs)]
        cache[q] = hs
        return hs

    def act_search(self, ctx: Ctx, idx: np.ndarray, now: int) -> np.ndarray:
        w = ctx.world
        rng = ctx.rng("popweb.search")
        c = lambda name: columns.col(w, TABLE, name)[idx]  # noqa: E731
        inter = c("interests").astype(np.float32)
        skills = c("skills").astype(np.float64)
        prefs = c("prefs").astype(np.float64)
        aband = c("abandoned")
        qs = np.array(semantics.query_words(inter, rng, 0.15 * (1.0 - 0.8 * skills[:, 0])))
        lim = int(self.config.get("search_limit", 6))
        tbl = w.state.tables["search_docs"]
        pages = w.state.find_ids("search_docs", "kind", "page")
        version = (len(pages), max((tbl[d]["seq"] for d in pages), default=0))
        uq, qinv = np.unique(qs, return_inverse=True)
        target = np.full(idx.size, NONE_I32, np.int64)
        n_res = np.zeros(idx.size, np.int16)
        rank = np.full(idx.size, -1, np.int16)
        for k, q in enumerate(uq):
            hs = self._search(ctx, str(q), lim, version)
            sel = np.flatnonzero(qinv == k)
            if not hs:
                continue
            rows_q = self._ensure_res(ctx, [h for h, _ in hs])
            okr = rows_q >= 0
            if not okr.any():
                continue
            rows_q = rows_q[okr]
            sv = np.stack([v for (h, v), o in zip(hs, okr) if o])
            sc = 2.0 * (inter[sel] @ sv.T) - 0.35 * np.arange(rows_q.size)[None, :] + \
                rng.gumbel(size=(sel.size, rows_q.size)) * (0.4 + 0.3 * prefs[sel, 1:2])
            blocked = (aband[sel][:, :, None] == rows_q[None, None, :]).any(axis=1)
            sc = np.where(blocked, -np.inf, sc)
            avail = np.isfinite(sc).any(axis=1)
            best = np.argmax(sc, axis=1)
            target[sel[avail]] = rows_q[best[avail]]
            rank[sel[avail]] = best[avail]
            n_res[sel] = (~blocked).sum(axis=1)
        via = np.full(idx.size, 3, np.int64)
        U, _ = self._visit(ctx, idx, target, via, np.full(idx.size, NONE_I32, np.int64), now, "search",
                           budget=core.session_budget(w, idx))
        U = np.where(target >= 0, U, -0.2)
        sk = c("skills").astype(np.float32)
        be = c("beliefs").astype(np.float32)
        sk[:, 0] = np.clip(sk[:, 0] + 0.01 * (U > 0.3), 0, 1)
        be[:, 1] = 0.85 * be[:, 1] + 0.15 * (U > 0.3)
        ref = artifacts.put_array_blob(ctx, {"ind": idx.astype(np.int32), "query": qinv.astype(np.int32),
                                             "n_results": n_res, "rank": rank, "res": target.astype(np.int32)})
        record_decisions(ctx, "popweb.searched", idx, {"skills": sk.astype(np.float16),
                                                       "beliefs": be.astype(np.float16)},
                         model_id="pop.search_learning", model_version="1",
                         fidelity_id="pop.decision.search_learning",
                         extra={"t": int(now), "queries": [str(q) for q in uq], "searches_ref": ref,
                                "with_results": int((n_res > 0).sum())})
        return U

    # ------------------------------------------------------------------ the forum (4.0 forum module)
    def _post_vecs(self, ctx: Ctx, posts: list) -> list[tuple[np.ndarray, float]]:
        """(vector in the interest coordinates, richness) of forum posts, from their text. Cached per process and
        world by the post's text (a pure function of it: the vector comes from the space's recorded representation,
        the richness from the text's intrinsic features), so a feed read by many individuals is represented once."""
        cache = ctx.world.runtime.setdefault("popweb.post_reps", {})
        texts = [(p["title"] + " " + p.get("body", "")).strip() or "post" for p in posts]
        miss = sorted({t for t in texts if t not in cache})
        stats = ctx.world.runtime.setdefault("popweb.stats", {})
        stats["post_rep_lookups"] = stats.get("post_rep_lookups", 0) + len(texts)
        stats["post_rep_computed"] = stats.get("post_rep_computed", 0) + len(miss)
        if miss:
            V = self._coords(ctx, miss)
            for j, t in enumerate(miss):
                stt = semantics.text_features(t)
                body_len = len(t)
                rich = math.sqrt(min(1.0, body_len / 1200.0)) * min(1.0, stt["distinct_ratio"] * 1.6) * \
                    (0.3 + 0.7 * semantics.legibility(stt))
                cache[t] = (V[j], rich)
            if len(cache) > 20000:
                for k in list(cache)[:5000]:
                    cache.pop(k, None)
        return [cache[t] for t in texts]

    def _coords(self, ctx: Ctx, texts: list[str]) -> np.ndarray:
        """Interest-space coordinates of texts (recorded text encoder, population.semantics)."""
        return semantics.project(semantics.text_vectors(ctx.world, list(texts), ctx=ctx), self.D)

    def _board_id(self, ctx, board: str) -> int:
        r = ctx.state.get("pop_boards", board)
        if r is not None:
            return int(r["id"])
        nid = len(ctx.state.tables["pop_boards"])
        ctx.emit("popweb.board", {"id": board, "n": nid}, vis="internal")
        return nid

    def act_forum(self, ctx: Ctx, idx: np.ndarray, now: int) -> np.ndarray:
        w = ctx.world
        st = ctx.state
        forum = w.modules["forum"]
        ds = w.runtime["pop.models"]
        rng = ctx.rng("popweb.forum")
        c = lambda name: columns.col(w, TABLE, name)[idx]  # noqa: E731
        n = idx.size
        inter = c("interests").astype(np.float32)
        prefs = c("prefs").astype(np.float64)
        skills = c("skills").astype(np.float64)
        goal_social = (c("goals") == 3).any(axis=1).astype(np.float64)
        comm = c("communities").copy()
        left = core.session_budget(w, idx).astype(np.float64)
        acts = {"read": 0, "reply": 0, "post": 0, "follow": 0, "join": 0, "vote": 0}
        U = np.full(n, -0.05)
        minutes = np.zeros(n)
        boards = sorted(st.tables["forum_boards"])
        bmat = self._coords(ctx, [(st.tables["forum_boards"][b]["title"] + " " +
                                   st.tables["forum_boards"][b].get("description", "")).strip() or b
                                  for b in boards]) if boards else np.zeros((0, self.D), np.float32)
        topics = semantics.nearest_topic(inter)
        # which post each reader reads: the shared public feed, or its own feed when it follows anything
        public = forum.feed(w, "pop:anyone", 20)
        read = [None] * n
        # readers who follow someone or a board read their own feed; the feed depends only on what one follows (and
        # the forum's state), so readers with the same follow set share one computation of it
        by_follows: dict[tuple, list] = {}
        for j in range(n):
            pj = f"p{int(idx[j])}"
            if pj not in st.tables["principals"]:
                continue
            fids = st.find_ids("forum_follows", "follower", pj)
            if fids:
                key = tuple(sorted(st.tables["forum_follows"][f]["target"] for f in fids))
                by_follows.setdefault(key, []).append(j)
        own_set = {j for js in by_follows.values() for j in js}
        stats = w.runtime.setdefault("popweb.stats", {})
        stats["feed_readers"] = stats.get("feed_readers", 0) + len(own_set)
        stats["feed_calls"] = stats.get("feed_calls", 0) + len(by_follows)
        groups = {None: [j for j in range(n) if j not in own_set]}
        for key, js in sorted(by_follows.items()):
            groups[key] = js
        for key, js in groups.items():
            feed = public if key is None else forum.feed(w, f"p{int(idx[js[0]])}", 20)
            if not feed or not js:
                continue
            pv = self._post_vecs(ctx, [st.tables["forum_posts"][p["post"]] for p in feed])
            M = np.stack([v for v, _ in pv])
            rich = np.array([r for _, r in pv])
            js = np.array(js)
            rel = inter[js] @ M.T
            pick = np.argmax(2.0 * rel + rng.gumbel(size=rel.shape) * 0.5, axis=1)
            for j, k in zip(js, pick):
                read[j] = (feed[k], float(rel[j == js][0][k]) if False else float(inter[j] @ M[k]), float(rich[k]))
        readers = np.array([j for j in range(n) if read[j] is not None], np.int64)
        # an empty forum: sociable individuals start a discussion about what they care about
        idle = np.array([j for j in range(n) if read[j] is None], np.int64)
        if idle.size and boards:
            start = idle[rng.random(idle.size) < 0.15 + 0.5 * prefs[idle, 6]]
            for j in start:
                b = boards[int(np.argmax(bmat @ inter[j]))]
                self._write(ctx, int(idx[j]), "post", b, int(topics[j]), rng, acts)
                U[j] = 0.1
        if readers.size:
            rel = np.array([read[j][1] for j in readers])
            rich = np.array([read[j][2] for j in readers])
            u = (0.2 + 1.0 * rel) * (0.4 + 0.6 * rich) + rng.normal(0, 0.1, readers.size)
            U[readers] = u
            minutes[readers] = np.minimum(left[readers], 1.0 + 4.0 * rich)
            acts["read"] = int(readers.size)
            F = {"page_utility": u, "sociability": prefs[readers, 6], "writing_skill": skills[readers, 1],
                 "goal_social": goal_social[readers]}
            reply = ds["participate"].sample(F, rng)
            follow = ds["follow_creator"].sample(F, rng)
            vote = (u > 0.5) & (rng.random(readers.size) < 0.3)
            all_comm = columns.col(w, TABLE, "communities")
            contacts = c("contacts").copy()
            trust = c("contact_trust").copy()
            for a, j in enumerate(readers):
                post = read[j][0]
                pr = st.tables["forum_posts"][post["post"]]
                i = int(idx[j])
                if reply[a]:
                    self._write(ctx, i, "reply", post["post"], int(topics[j]), rng, acts)
                if vote[a]:
                    self._call(ctx, i, "forum.vote", {"post": post["post"], "value": 1}, acts, "vote")
                if follow[a] and pr["author"] != f"p{i}":
                    if self._call(ctx, i, "forum.follow", {"target": pr["author"]}, acts, "follow"):
                        au = ids.index(pr["author"])
                        if au is not None and au not in contacts[j]:
                            k = int(np.argmin(np.where(contacts[j] >= 0, trust[j].astype(np.float32), -1.0)))
                            contacts[j, k], trust[j, k] = au, 0.6
            # joining the community the post belongs to
            bids = np.array([self._board_id(ctx, read[j][0]["board"]) for j in readers])
            cand = ~_member(comm[readers], bids)
            ct = columns.col(w, TABLE, "contacts")[idx[readers]].astype(np.int64)
            share = ((all_comm[np.clip(ct, 0, None)] == bids[:, None, None]).any(axis=2) & (ct >= 0)).mean(axis=1)
            brel = np.array([float(bmat[boards.index(read[j][0]["board"])] @ inter[j]) if read[j][0]["board"] in boards
                             else 0.0 for j in readers])
            Fj = {"board_relevance": brel, "sociability": prefs[readers, 6], "contacts_in_community": share,
                  "goal_social": goal_social[readers]}
            join = ds["join_community"].sample(Fj, rng) & cand
            for a in np.flatnonzero(join):
                j = readers[a]
                if self._call(ctx, int(idx[j]), "forum.follow", {"target": f"board:{read[j][0]['board']}"}, acts,
                              "join"):
                    comm[j, 1:] = comm[j, :-1]
                    comm[j, 0] = bids[a]
        # sites named in what was read become suggestions
        sug = c("suggest").astype(np.int64)
        ssrc = c("suggest_src").astype(np.int64)
        sfrom = c("suggest_from").astype(np.int64)
        hs = [(j, read[j][0]["hosts"][0]) for j in readers if read[j][0]["hosts"] and sug[j] < 0] if readers.size else []
        if hs:
            rr = self._ensure_res(ctx, [h for _, h in hs])
            for (j, _), r in zip(hs, rr):
                if r >= 0:
                    sug[j], ssrc[j], sfrom[j] = r, 3, NONE_I32
        sk = c("skills").astype(np.float32)
        if readers.size:
            sk[readers, 5] = np.clip(sk[readers, 5] + 0.005, 0, 1)
        if minutes.sum() > 0:
            mi = np.flatnonzero(minutes > 0)
            attention.spend(ctx, idx[mi], minutes[mi], activity="forum")
        vals = {"communities": comm, "suggest": sug.astype(np.int32), "suggest_src": ssrc.astype(np.uint8),
                "suggest_from": sfrom.astype(np.int32), "skills": sk.astype(np.float16)}
        if readers.size and not np.array_equal(contacts, c("contacts")):
            vals.update({"contacts": contacts, "contact_trust": trust})
        record_decisions(ctx, "popweb.forum", idx, vals, model_id="pop.forum", model_version="1",
                         fidelity_id="pop.decision.forum", extra={"t": int(now), **acts})
        return U

    def _call(self, ctx, i, tool, args, acts, label) -> bool:
        p = ids.ensure_principal(ctx, i)
        try:
            ctx.call(tool, args, as_principal=p)
            acts[label] += 1
            return True
        except CrucibleError:
            return False

    def _write(self, ctx, i, kind, target, topic, rng, acts) -> None:
        """Write a reply or a new post about its topic (population.writing, from a real corpus), naming a site it
        likes."""
        from . import writing
        w = ctx.world
        ks = columns.col(w, TABLE, "known_sat")[i]
        kn = columns.col(w, TABLE, "known")[i]
        fav = ""
        if (kn >= 0).any() and float(ks.max()) > 0.55:
            r = int(kn[int(np.argmax(np.where(kn >= 0, ks, -9)))])
            hosts = self._hosts(w)
            if r < len(hosts):
                fav = f" I found {hosts[r]}.crucible useful."
        skill = float(columns.col(w, TABLE, "skills")[i, 1])
        title, body = writing.compose(rng, topic, int(rng.integers(1, 4)), skill)
        body = body + fav
        if kind == "reply":
            self._call(ctx, i, "forum.reply", {"post": target, "body": body}, acts, "reply")
        else:
            self._call(ctx, i, "forum.post", {"board": target, "title": title, "body": body}, acts, "post")

    # ------------------------------------------------------------------ owner analytics
    @endpoint("visits", S.obj({"host": S.s_str(maxLength=31), "days": S.s_int(minimum=1, maximum=60)}, ["host"]),
              read_only=True, cost_s=1)
    def ep_visits(self, ctx, host, days=7):
        """Visits to a site you operate by members of the population (totals so far): visits, first visits,
        returns, purchases, revenue. Only the operator of the site can read them."""
        from ..services.identity import ident
        st = ctx.state
        h = st.get("hosts", host)
        if h is None:
            raise NotFound("no such host")
        ident(ctx.world).authorize(ctx, h["owner"], "analytics.read")
        r = st.get("pop_res", host)
        if r is None:
            return {"host": host, "total": {}}
        row = r["row"]
        g = lambda k: int(columns.col(ctx.world, RES_TABLE, k)[row])  # noqa: E731
        return {"host": host, "total": {k: g(k) for k in ("visits", "first_visits", "return_visits", "purchases",
                                                          "revenue_cents")}}

    # ------------------------------------------------------------------ appliers
    @applier("popweb.resource")
    def ap_resource(self, ev):
        p = ev.payload
        self.world.state.put("pop_res", p["id"], {"row": p["row"], "owner": p["owner"], "t": ev.t})

    @applier("popweb.page")
    def ap_page(self, ev):
        p = ev.payload
        self.world.state.put("pop_pages", p["key"], {**p, "t": ev.t})

    @applier("popweb.pagerow")
    def ap_pagerow(self, ev):
        self.world.state.put("pop_pagerow", ev.payload["key"], {"row": ev.payload["row"], "t": ev.t})

    @applier("popweb.board")
    def ap_board(self, ev):
        self.world.state.put("pop_boards", ev.payload["id"], {"id": ev.payload["n"]})

    def check_invariants(self, world):
        errs = []
        st = world.state
        R = columns.rows(world, RES_TABLE)
        if len(st.tables["pop_res"]) != R:
            errs.append(f"pop_res has {len(st.tables['pop_res'])} records for {R} resource rows")
        rows = sorted(r["row"] for r in st.tables["pop_res"].values())
        if rows != list(range(len(rows))):
            errs.append("resource rows are not 0..R-1")
        if R:
            v = columns.col(world, RES_TABLE, "visits")
            fv = columns.col(world, RES_TABLE, "first_visits")
            rv = columns.col(world, RES_TABLE, "return_visits")
            if not np.array_equal(v, fv + rv):
                errs.append("visits != first + return visits")
        return errs
