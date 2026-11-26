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
from .schema import NONE_I32, RES_TABLE, TABLE

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


def _member(a: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Row-wise: is v[i] in a[i, :]? (a [n, k], v [n])."""
    return (a == v[:, None]).any(axis=1) if a.size else np.zeros(v.shape[0], bool)


@register_module
class PopWeb(Module):
    """Resource discovery, visits, use, return, abandonment, purchases; activities `web`, `search`, `forum`.
    Config: {page_ttl_s: 10800, max_depth: 3, genesis_known: 2, search_limit: 6, forum: true}."""
    name = "popweb"
    host = "popweb"
    version = "1"
    depends = ("pop", "space", "identity", "bank", "code", "cloud", "net", "search", "artifacts", "cols")

    def install(self, world):
        pop = world.modules["pop"]
        self.d = pop.d
        self.D = pop.d["D"]
        columns.declare(world, RES_TABLE, 0, schema.res_columns(pop.d))
        st = world.state
        st.declare_table("pop_res", index=("owner",))
        st.declare_table("pop_pages", index=("host",))
        st.declare_table("pop_boards")
        self.ttl = int(self.config.get("page_ttl_s", 3 * 3600))
        self.max_depth = int(self.config.get("max_depth", 3))
        core.register_activity(world, "web", self.act_web, owner=self.name, prior=1.0, goals=(1, 2, 4))
        core.register_activity(world, "search", self.act_search, owner=self.name, prior=0.5, goals=(1, 2))
        self.forum_on = bool(self.config.get("forum", True)) and any(m["name"] == "forum" for m in world.spec.modules)
        if self.forum_on:
            core.register_activity(world, "forum", self.act_forum, owner=self.name, prior=0.1, goals=(3, 1))

    def on_load(self, world):
        for k in ("popweb.posts", "popweb.hosts", "popweb.search"):
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
        stt = {"uniq": semantics.text_features(page_text)["distinct_ratio"]}
        vec = self._coords(ctx, [page_text])[0] if status == 200 and page_text else np.zeros(self.D, np.float32)
        dep_id, why = self._dep_state(st, host)
        out = {"key": key, "host": host, "path": path, "status": status, "digest": text.body_digest(body),
               "dep": dep_id, "why": why, "title": parsed["title"], "vec": [round(float(x), 3) for x in vec],
               "chars": len(parsed["text"]), "uniq": round(float(stt["uniq"]), 3), "paths": parsed["paths"][:30],
               "hosts": [h for h in parsed["hosts"] if h != host][:12], "prices": parsed["prices"][:10],
               "kind": kind_of(parsed["title"]), "visitor": p}
        ctx.emit("popweb.page", out, vis="internal")
        return st.get("pop_pages", key)

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
        V = 0.1 + 1.25 * ks + 1.4 * match + 0.25 * prefs[:, 5:6] * np.log1p(kc) - 0.5 * recency + \
            0.6 * prefs[:, 4:5] * social
        V = np.where(valid, V, -np.inf)
        sug = c("suggest").astype(np.int64)
        sfrom = c("suggest_from").astype(np.int64)
        src = c("suggest_src").astype(np.int64)
        trust = np.full(n, 0.5)
        if (sfrom >= 0).any():
            slot = contacts == sfrom[:, None]
            tr = c("contact_trust").astype(np.float64)
            trust = np.where(slot.any(axis=1), (tr * slot).sum(axis=1) / np.maximum(slot.sum(axis=1), 1), 0.5)
        best = np.where(valid.any(axis=1), np.max(np.where(valid, ks, -1.0), axis=1), 0.0)
        explore = -0.15 + 0.8 * prefs[:, 1] + 0.5 * (1.0 - np.clip(best, 0, 1)) + \
            np.where(sug >= 0, 0.3 + 0.6 * np.where(src == 2, trust, beliefs[:, 5]), 0.4 * beliefs[:, 0] - 0.2)
        scores = np.concatenate([V, explore[:, None]], axis=1)
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
               activity: str) -> np.ndarray:
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
        bud = c("attention_budget_min").astype(np.float64)
        left = c("attention_left_min").astype(np.float64)
        path = ["/"] * n
        active = rows >= 0
        u_sum = np.zeros(n)
        pages = np.zeros(n, np.int64)
        okp = np.zeros(n, np.int64)
        minutes = np.zeros(n)
        consumed = np.zeros((n, D), np.float32)
        bought = np.zeros(n, bool)
        paid = np.zeros(n, np.int64)
        evaluated = np.zeros(n, bool)
        match0 = np.zeros(n, np.float32)     # interest match with the entry page, at the time of the decision
        new_seen = {}            # row -> latest entry-page record (refreshes the res table)
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
            P_prods = [[p for p in r["paths"] if p.startswith("/p/")] for r in recs]
            P_arts = [[p for p in r["paths"] if p.startswith("/a/") or p == "/shop"] for r in recs]
            P_price = np.array([(r["prices"][0] if r["prices"] and r["path"].startswith("/p/") else -1) for r in recs])
            for k, r in enumerate(recs):
                if r["path"] == "/":
                    new_seen[int(rows[act[first_i[k]]])] = r
            ok, V, chars, uniq = P_ok[inv], P_V[inv], P_chars[inv], P_uniq[inv]
            rel = np.einsum("nd,nd->n", inter[act], V.astype(np.float32)) * ok
            if depth == 0:
                match0[act] = rel
            rich = np.sqrt(np.minimum(1.0, chars / 2000.0)) * np.clip(uniq * 1.6, 0, 1)
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
            # a visitor on a product page decides whether to buy it (through the site's checkout)
            price = P_price[inv]
            on_prod = np.flatnonzero(ok & (price > 0) & ~bought[act])
            if on_prod.size:
                j = act[on_prod]
                evaluated[j] = True
                pr = price[on_prod].astype(np.float64)
                wallet = core.spendable(w, idx[j]).astype(np.float64)
                wtp = 2500.0 * wtp_s[j] * (1.0 + 1.5 * np.maximum(rel[on_prod], 0.0))
                F = {"price_ratio": np.clip((wtp - pr) / wtp, -3, 1), "page_relevance": rel[on_prod],
                     "goal_buy": buyer[j], "wallet_cover_log": np.clip(np.log(np.maximum(wallet, 1) / pr), -5, 5),
                     "price_sensitivity": prefs[j, 0], "resource_satisfaction": self._sat_of(c, j, rows[j]),
                     "belief_shops": beliefs[j, 3], "impulsiveness": prefs[j, 7]}
                buy = ds["purchase"].sample(F, rng) & (wallet >= pr)
                for jj, a, pp in zip(j[buy], on_prod[buy], pr[buy]):
                    got = self._purchase(ctx, int(idx[jj]), hosts_all[rows[jj]], path[jj], int(pp))
                    if got:
                        bought[jj] = True
                        paid[jj] = got
                        u_sum[jj] += 0.4 + 0.4 * float(np.clip((wtp[a == on_prod][0] - pp) / wtp[a == on_prod][0], -1, 1))
            # continue reading?  (evaluate a product, read another article)
            Fc = {"page_utility": u, "attention_ratio": left[act] / np.maximum(bud[act], 1e-3),
                  "quality_sensitivity": qs, "novelty_seeking": nseek, "page_richness": rich}
            cont = ds["continue"].sample(Fc, rng) & ok & (depth < self.max_depth)
            Fe = {"page_relevance": rel, "goal_buy": buyer[act], "impulsiveness": prefs[act, 7],
                  "shopping_skill": skills[act, 3], "attention_ratio": left[act] / np.maximum(bud[act], 1e-3)}
            want = ds["evaluate_product"].sample(Fe, rng)
            r_pick = rng.random(act.size)
            active[:] = False
            for a in np.flatnonzero(cont):
                k = inv[a]
                lst = P_prods[k] if (want[a] and P_prods[k]) else P_arts[k]
                if lst:
                    nxt = lst[int(r_pick[a] * len(lst))]
                    j = act[a]
                    if nxt != path[j]:
                        path[j] = nxt
                        active[j] = True
        U = np.where(pages > 0, u_sum / np.maximum(pages, 1), -0.1)
        self._record(ctx, idx, rows, via, frm, U, pages, okp, minutes, consumed, bought, paid, evaluated, new_seen,
                     now, activity, rng, match0)
        ad = w.modules.get("pop_adaptive")
        if ad is not None:
            ad.reward(ctx, idx, U)          # adaptive participants learn from the utility they experienced
        return U

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
                now, activity, rng, match0):
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
            purchased = c("purchased").copy()
            purchased[b_ix, 1:] = purchased[b_ix, :-1]
            purchased[b_ix, 0] = rows[b_ix]
            goals = c("goals").copy()
            goals[b_ix, 0] = np.where(goals[b_ix, 0] == 2, 4, goals[b_ix, 0])        # a buying goal is met
            vals.update({"purchased": purchased, "goals": goals,
                         "n_purchases": (c("n_purchases") + bought).astype(np.int32),
                         "spent_cents": (c("spent_cents") + paid).astype(np.int64)})
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
        columns.batch(ctx, TABLE, f"popweb.{activity}", idx, vals, mode="set",
                      extra={"t": int(now), "visits": int(v.size), "first": int(first.sum()),
                             "abandoned": int(abandoned.sum()), "bought": int(bought.sum()), "visits_ref": ref})
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
                    columns.batch(ctx, TABLE, "popweb.recommend", to[o], {
                        "suggest": rs_[o].astype(np.int32), "suggest_from": fr[o].astype(np.int32),
                        "suggest_src": np.full(to.size, 2, np.uint8)}, extra={"t": int(now), "n": int(to.size)})

    # ------------------------------------------------------------------ activities
    def act_web(self, ctx: Ctx, idx: np.ndarray, now: int) -> np.ndarray:
        rng = ctx.rng("popweb.choose")
        rows, via, frm = self._choose(ctx, idx, now, rng)
        return self._visit(ctx, idx, rows, via, frm, now, "web")

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
        U = self._visit(ctx, idx, target, via, np.full(idx.size, NONE_I32, np.int64), now, "search")
        U = np.where(target >= 0, U, -0.2)
        sk = c("skills").astype(np.float32)
        be = c("beliefs").astype(np.float32)
        sk[:, 0] = np.clip(sk[:, 0] + 0.01 * (U > 0.3), 0, 1)
        be[:, 1] = 0.85 * be[:, 1] + 0.15 * (U > 0.3)
        ref = artifacts.put_array_blob(ctx, {"ind": idx.astype(np.int32), "query": qinv.astype(np.int32),
                                             "n_results": n_res, "rank": rank, "res": target.astype(np.int32)})
        columns.batch(ctx, TABLE, "popweb.searched", idx, {"skills": sk.astype(np.float16),
                                                           "beliefs": be.astype(np.float16)},
                      extra={"t": int(now), "queries": [str(q) for q in uq], "searches_ref": ref,
                             "with_results": int((n_res > 0).sum())})
        return U

    # ------------------------------------------------------------------ the forum (4.0 forum module)
    def _post_vecs(self, ctx: Ctx, posts: list) -> list[tuple[np.ndarray, float]]:
        """(vector in the interest coordinates, richness) of forum posts, from their text."""
        texts = [(p["title"] + " " + p.get("body", "")).strip() or "post" for p in posts]
        V = self._coords(ctx, texts) if texts else np.zeros((0, self.D), np.float32)
        out = []
        for j, p in enumerate(posts):
            stt = semantics.text_features(texts[j])
            rich = math.sqrt(min(1.0, len(p.get("body", "")) / 1200.0)) * min(1.0, stt["distinct_ratio"] * 1.6) * \
                (0.3 + 0.7 * semantics.legibility(stt))
            out.append((V[j], rich))
        return out

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
        left = c("attention_left_min").astype(np.float64)
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
        own = [j for j in range(n) if f"p{int(idx[j])}" in st.tables["principals"]
               and st.find_ids("forum_follows", "follower", f"p{int(idx[j])}")]
        groups = {None: [j for j in range(n) if j not in set(own)]}
        for j in own:
            groups[j] = [j]
        for key, js in groups.items():
            feed = public if key is None else forum.feed(w, f"p{int(idx[key])}", 20)
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
        columns.batch(ctx, TABLE, "popweb.forum", idx, vals, extra={"t": int(now), **acts})
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
