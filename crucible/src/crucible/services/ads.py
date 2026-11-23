"""Search advertising (host `ads`): paid distribution for products and services.

An advertiser (a person or organisation) runs campaigns: keywords, an optional target document kind, a bid per
impression, a daily budget and a landing document (title/text/hosts/ref) shown as a sponsored search result.
Creating a campaign requires `bank.spend` authority over the paying account; that authority is the standing
order under which impressions are charged later (inside other principals' search commands).

* `world.hooks["sponsored"](world, q, kind, limit)` (installed here) returns the highest-bidding active
  campaigns whose keywords match a query token, whose target kind fits and whose daily budget is not exhausted;
  at most one campaign per advertiser per query.
* `record_impressions(ctx, campaign_ids, n=1)` charges `bid * n` immediately via `bank.move` from the
  campaign account to the operator account (`config["operator_account"]`, default `acct_fees`). A campaign that
  cannot pay is paused (`reason="insufficient_funds"`) and the impression is not served (the search endpoint
  drops sponsored results that were not charged). Returns the campaign ids actually charged.
* `search_impressions(ctx, campaign_ids)` is what a search by a participant uses: a campaign pays for at most ONE
  impression per searcher per simulated day. The searcher is the person behind the search (a deployed service
  counts as the people who control its operator), so repeating a query - by hand, from a script, or from any
  number of one's own services - cannot run down somebody else's advertising budget; the ad is still shown on
  repeats. Who has already been charged is read from the ledger (`bank.txn.by` of the impression charges), so the
  rule needs no state of its own. Impressions served to the simulated consumer population are charged in bulk by
  the demand process through `record_impressions` and are not affected.
* A campaign created under a spending grant that has a limit remembers the grant: every impression is checked
  against it and recorded on it, and the campaign pauses when the limit is used up.
* A daily process (`ads.daily`) resets `spent_today`.

Invariants: per campaign, `total_spent == impressions * bid_cents == sum of its ad_impressions bank txns`,
`spent_today <= daily_budget_cents`; keyword rows reference existing campaigns.
"""
from __future__ import annotations

from ..core import schema as S
from ..core.errors import Conflict, CrucibleError, Forbidden, InsufficientFunds, InvalidAction, NotFound
from ..core.kernel import SYSTEM, Ctx, Module, applier, endpoint, process, register_module
from .identity import ident
from .search import HOST_NAME, KIND_RE, is_host, tokenize

DAY = 86400
MAX_BID = 10_000
MAX_BUDGET = 10**9
MAX_KEYWORDS = 20
MAX_CAMPAIGNS = 100
TXN_KIND = "ad_impressions"


@register_module
class Ads(Module):
    name = "ads"
    host = "ads"
    depends = ("identity", "bank")

    def install(self, world):
        st = world.state
        st.declare_table("ads_campaigns", index=("advertiser", "status"))
        st.declare_table("ads_kw", index=("kw", "campaign"))
        world.hooks["sponsored"] = self.sponsored

    def genesis(self, ctx):
        ctx.schedule(DAY, "ads.daily", {}, prio=10, period=DAY, key="ads.daily")

    @property
    def operator_account(self) -> str:
        return self.config.get("operator_account", "acct_fees")

    # ------------------------------------------------------------------ hook: sponsored results
    def sponsored(self, world, q: str, kind: str | None, limit: int) -> list[dict]:
        st = world.state
        hits: dict[str, int] = {}
        for t in sorted(set(tokenize(q))):
            for cid in st.find_ids("ads_kw", "kw", t):
                c = st.tables["ads_kw"][cid]["campaign"]
                hits[c] = hits.get(c, 0) + 1
        rows = []
        for cid in sorted(hits):
            c = st.tables["ads_campaigns"][cid]
            if c["status"] != "active":
                continue
            if kind is not None and c["kind"] not in (None, kind):
                continue
            if c["spent_today"] + c["bid_cents"] > c["daily_budget_cents"]:
                continue
            rows.append((-c["bid_cents"], -hits[cid], c["created"], cid))
        rows.sort()
        out, seen = [], set()
        for _, _, _, cid in rows:
            c = st.tables["ads_campaigns"][cid]
            if c["advertiser"] in seen:
                continue
            seen.add(c["advertiser"])
            land = c["landing"]
            out.append({"id": f"ad:{cid}", "campaign": cid, "kind": c["kind"] or "ad", "title": land["title"],
                        "text": land["text"], "hosts": list(land["hosts"]), "ref": land["ref"],
                        "owner": c["advertiser"], "t": c["created"], "bid_cents": c["bid_cents"]})
            if len(out) >= limit:
                break
        return out

    # ------------------------------------------------------------------ internal API
    def record_impressions(self, ctx: Ctx, campaign_ids, n: int = 1) -> list[str]:
        """Charge `bid * n` per campaign now (capped by the remaining daily budget); pause campaigns that cannot
        pay. Returns the ids of campaigns that were charged (their impressions were served)."""
        st = ctx.state
        bank = ctx.world.modules["bank"]
        op = self.operator_account
        if n < 1 or not st.has("accounts", op):
            return []
        charged: list[str] = []
        for cid in campaign_ids:
            c = st.get("ads_campaigns", cid)
            if c is None or c["status"] != "active" or cid in charged:
                continue
            bid = c["bid_cents"]
            k = min(int(n), (c["daily_budget_cents"] - c["spent_today"]) // bid)
            if k <= 0:
                continue
            acct = st.get("accounts", c["account"])
            if acct is None or acct["status"] != "open":
                self._pause(ctx, cid, c, "account_closed")
                continue
            k = min(k, bank.available(st, c["account"]) // bid)  # type: ignore[attr-defined]
            if k <= 0:
                self._pause(ctx, cid, c, "insufficient_funds")
                continue
            grant = c.get("grant")
            if grant:
                k = min(k, ident(ctx.world).grant_room(ctx, grant) // bid)
                if k <= 0:
                    self._pause(ctx, cid, c, "spending_limit")
                    continue
            try:
                tid = bank.move(ctx, c["account"], op, k * bid, TXN_KIND, f"ad impressions {cid}", ref=cid)  # type: ignore[attr-defined]
            except CrucibleError:   # bank.post validates before emitting: nothing to undo
                self._pause(ctx, cid, c, "payment_failed")
                continue
            if grant:
                ident(ctx.world).grant_spend(ctx, grant, k * bid)
            ctx.emit("ads.impressions", {"id": cid, "n": k, "amount": k * bid, "txn": tid}, vis="internal")
            charged.append(cid)
        return charged

    def viewers(self, state, actor: str) -> set[str]:
        """Every principal that is the same audience as `actor`: the persons behind it (itself; for a service the
        owners and admins of its operator) and the services of everything those persons control."""
        people: set[str] = set()
        rec = state.get("principals", actor)
        op = rec["info"].get("operator") if rec is not None and rec["kind"] == "service" else actor
        if isinstance(op, str) and op.startswith("o:"):
            people.update(m["person"] for m in state.find("members", "org", op) if m["role"] in ("owner", "admin"))
        elif isinstance(op, str):
            people.add(op)
        out = set(people) | {actor}
        if "hosts" in state.tables:
            owners = set(people)
            for p in people:
                owners.update(m["org"] for m in state.find("members", "person", p) if m["role"] in ("owner", "admin"))
            for o in sorted(owners):
                out.update(f"svc:{h}" for h in state.find_ids("hosts", "owner", o))
        return out

    def seen_today(self, state, cid: str, viewers: set[str], now: int) -> bool:
        """Whether one of `viewers` has already been charged an impression of this campaign this simulated day."""
        idx = state.indexes.get(("txns", "by"))
        if idx is None:
            return False
        day0 = (now // DAY) * DAY
        txns = state.tables["txns"]
        for v in viewers:
            for tid in idx.get(v, ()):
                tx = txns[tid]
                if tx["kind"] == TXN_KIND and tx["ref"] == cid and tx["t"] >= day0:
                    return True
        return False

    def search_impressions(self, ctx: Ctx, campaign_ids) -> list[str]:
        """Impressions for one search by `ctx.actor`: returns the campaigns that may be shown - those charged now
        (first impression of the day for this searcher) and those this searcher has already been charged for
        today (shown again at no charge)."""
        st = ctx.state
        if ctx.actor == SYSTEM:                      # an operator's own query: charged like any impression
            return self.record_impressions(ctx, campaign_ids, 1)
        who = self.viewers(st, ctx.actor)
        shown: list[str] = []
        for cid in campaign_ids:
            if cid in shown:
                continue
            if self.seen_today(st, cid, who, ctx.now):
                shown.append(cid)
            elif self.record_impressions(ctx, [cid], 1):
                shown.append(cid)
        return shown

    def _pause(self, ctx: Ctx, cid: str, c, reason: str) -> None:
        ctx.emit("ads.status", {"id": cid, "status": "paused", "reason": reason, "by": ctx.actor},
                 vis=(c["advertiser"],))

    def _campaign(self, ctx, cid: str):
        c = ctx.state.get("ads_campaigns", cid)
        if c is None:
            raise NotFound(f"no campaign {cid}")
        return c

    # ------------------------------------------------------------------ endpoints
    @endpoint("create_campaign", S.obj({
        "account": S.s_str("bank account that pays for impressions (needs bank.spend on its owner)", maxLength=40),
        "keywords": S.s_arr(S.s_str(maxLength=40), "query words that trigger the ad", minItems=1,
                            maxItems=MAX_KEYWORDS),
        "bid_cents": S.s_int("price per impression in cents", minimum=1, maximum=MAX_BID),
        "daily_budget_cents": S.s_int("maximum spend per simulated day", minimum=1, maximum=MAX_BUDGET),
        "title": S.s_str("headline of the sponsored result", minLength=1, maxLength=120),
        "text": S.s_str("body of the sponsored result", maxLength=500),
        "hosts": S.s_arr(S.s_str(pattern=HOST_NAME, maxLength=41), "service hosts the ad links to", maxItems=5),
        "ref": S.s_str("what the ad points to (listing id, url, ...)", maxLength=120),
        "kind": S.s_str("only show for searches of this kind (listing, service, job, ...)", pattern=KIND_RE,
                        maxLength=24)},
        ["account", "keywords", "bid_cents", "daily_budget_cents", "title"]), cost_s=60)
    def ep_create(self, ctx, account, keywords, bid_cents, daily_budget_cents, title, text="", hosts=(), ref=None,
                  kind=None):
        """Start a search-ad campaign. Searchers whose query contains one of the keywords see the ad above organic
        results (highest bid first, one ad per advertiser) and learn the hosts it links to. Each impression is
        charged to `account` immediately; the campaign pauses itself when the account cannot pay."""
        st = ctx.state
        bank = ctx.world.modules["bank"]
        owner = bank.account_owner(st, account)  # type: ignore[attr-defined]
        grant = ident(ctx.world).authorize_standing(ctx, owner, account)
        if grant and ident(ctx.world).grant_room(ctx, grant) < bid_cents:
            raise InsufficientFunds("your spending limit on this account does not cover a single impression",
                                    account=account)
        if st.tables["accounts"][account]["kind"] == "system":
            raise InvalidAction("system accounts cannot advertise")
        if daily_budget_cents < bid_cents:
            raise InvalidAction("daily budget is smaller than one impression")
        kws = sorted({t for k in keywords for t in tokenize(k)})
        if not kws:
            raise InvalidAction("keywords contain no searchable words")
        if len(kws) > MAX_KEYWORDS:
            raise InvalidAction(f"at most {MAX_KEYWORDS} keyword tokens")
        hs = sorted(set(hosts))
        for h in hs:
            if not is_host(ctx.world, h):
                raise InvalidAction(f"unknown host {h}")
        if len(st.find_ids("ads_campaigns", "advertiser", owner)) >= MAX_CAMPAIGNS:
            raise Conflict("campaign limit reached")
        if bank.available(st, account) < bid_cents:  # type: ignore[attr-defined]
            raise InsufficientFunds("account cannot pay for a single impression", account=account)
        cid = st.new_id("adc")
        ctx.emit("ads.campaign", {"id": cid, "advertiser": owner, "account": account, "keywords": kws, "kind": kind,
                                  "bid_cents": bid_cents, "daily_budget_cents": daily_budget_cents,
                                  "landing": {"title": title, "text": text, "hosts": hs, "ref": ref},
                                  "by": ctx.actor, **({"grant": grant} if grant else {})}, vis=(owner, ctx.actor))
        return {"campaign": cid, "keywords": kws}

    @endpoint("pause", S.obj({"campaign": S.s_str(maxLength=40)}, ["campaign"]), cost_s=5)
    def ep_pause(self, ctx, campaign):
        """Pause a campaign (needs bank.spend on its account's owner)."""
        c = self._campaign(ctx, campaign)
        ident(ctx.world).authorize(ctx, c["advertiser"], "bank.spend", scope=c["account"])
        if c["status"] == "paused":
            raise Conflict("already paused")
        ctx.emit("ads.status", {"id": campaign, "status": "paused", "reason": "manual", "by": ctx.actor},
                 vis=(c["advertiser"], ctx.actor))
        return {"ok": True}

    @endpoint("resume", S.obj({"campaign": S.s_str(maxLength=40)}, ["campaign"]), cost_s=5)
    def ep_resume(self, ctx, campaign):
        """Resume a paused campaign (the account must be able to pay for an impression)."""
        c = self._campaign(ctx, campaign)
        grant = ident(ctx.world).authorize_standing(ctx, c["advertiser"], c["account"])
        if grant and grant != c.get("grant"):
            raise Forbidden("a spending grant with a limit can only resume campaigns that were created under it")
        if c["status"] == "active":
            raise Conflict("already active")
        bank = ctx.world.modules["bank"]
        a = ctx.state.get("accounts", c["account"])
        if a is None or a["status"] != "open":
            raise InvalidAction("campaign account is not open")
        if bank.available(ctx.state, c["account"]) < c["bid_cents"]:  # type: ignore[attr-defined]
            raise InsufficientFunds("account cannot pay for a single impression", account=c["account"])
        ctx.emit("ads.status", {"id": campaign, "status": "active", "reason": None, "by": ctx.actor},
                 vis=(c["advertiser"], ctx.actor))
        return {"ok": True}

    @endpoint("stats", S.obj({"campaign": S.s_str("one campaign (default: all of yours)", maxLength=40)}),
              read_only=True)
    def ep_stats(self, ctx, campaign=None):
        """Campaign performance: status, impressions, spend today and in total."""
        if campaign is not None:
            c = self._campaign(ctx, campaign)
            ident(ctx.world).authorize(ctx, c["advertiser"], "analytics.read")
            return [_row(campaign, c)]
        out = []
        for owner in sorted(ctx.world.principals_of(ctx.actor)):
            for cid in ctx.state.find_ids("ads_campaigns", "advertiser", owner):
                out.append(_row(cid, ctx.state.tables["ads_campaigns"][cid]))
        return out

    # ------------------------------------------------------------------ process
    @process("daily")
    def pr_daily(self, ctx):
        ids = [cid for cid, c in ctx.state.scan("ads_campaigns") if c["spent_today"]]
        if ids:
            ctx.emit("ads.reset", {"day": ctx.now // DAY, "ids": ids}, vis="internal")

    # ------------------------------------------------------------------ appliers
    @applier("ads.campaign")
    def ap_campaign(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("ads_campaigns", p["id"], {"advertiser": p["advertiser"], "account": p["account"],
                                             "keywords": list(p["keywords"]), "kind": p["kind"],
                                             "bid_cents": p["bid_cents"],
                                             "daily_budget_cents": p["daily_budget_cents"], "status": "active",
                                             "reason": None, "spent_today": 0, "total_spent": 0, "impressions": 0,
                                             "landing": dict(p["landing"]), "created": ev.t, "by": p["by"],
                                             **({"grant": p["grant"]} if p.get("grant") else {})})
        for kw in p["keywords"]:
            st.insert("ads_kw", f"{kw}|{p['id']}", {"kw": kw, "campaign": p["id"]})

    @applier("ads.status")
    def ap_status(self, ev):
        self.world.state.patch("ads_campaigns", ev.payload["id"], status=ev.payload["status"],
                               reason=ev.payload["reason"])

    @applier("ads.impressions")
    def ap_impressions(self, ev):
        p = ev.payload
        c = self.world.state.tables["ads_campaigns"][p["id"]]
        self.world.state.patch("ads_campaigns", p["id"], spent_today=c["spent_today"] + p["amount"],
                               total_spent=c["total_spent"] + p["amount"], impressions=c["impressions"] + p["n"])

    @applier("ads.reset")
    def ap_reset(self, ev):
        for cid in ev.payload["ids"]:
            self.world.state.patch("ads_campaigns", cid, spent_today=0)

    # ------------------------------------------------------------------ invariants & observation
    def check_invariants(self, world):
        st = world.state
        errs = []
        paid: dict[str, int] = {}
        if "txns" in st.tables:
            for tid in st.find_ids("txns", "kind", TXN_KIND):
                tx = st.tables["txns"][tid]
                paid[tx["ref"]] = paid.get(tx["ref"], 0) + sum(d for _, d in tx["postings"] if d > 0)
        for cid, c in st.tables["ads_campaigns"].items():
            if c["total_spent"] != paid.get(cid, 0):
                errs.append(f"campaign {cid} spent {c['total_spent']} but was charged {paid.get(cid, 0)}")
            if c["total_spent"] != c["impressions"] * c["bid_cents"]:
                errs.append(f"campaign {cid} spend != impressions * bid")
            if not 0 <= c["spent_today"] <= c["daily_budget_cents"]:
                errs.append(f"campaign {cid} spent_today {c['spent_today']} outside [0, daily budget]")
            if c["status"] not in ("active", "paused"):
                errs.append(f"campaign {cid} has bad status {c['status']}")
        for kid, k in st.tables["ads_kw"].items():
            if k["campaign"] not in st.tables["ads_campaigns"]:
                errs.append(f"keyword row {kid} references a missing campaign")
        for cid in paid:
            if cid not in st.tables["ads_campaigns"]:
                errs.append(f"ad charges reference unknown campaign {cid}")
        return errs

    def observe(self, world, actor):
        st = world.state
        active = spent = 0
        for owner in sorted(world.principals_of(actor)):
            for cid in st.find_ids("ads_campaigns", "advertiser", owner):
                c = st.tables["ads_campaigns"][cid]
                active += c["status"] == "active"
                spent += c["spent_today"]
        return {"active_campaigns": active, "spent_today": spent} if active or spent else None


def _row(cid: str, c) -> dict:
    return {"campaign": cid, "advertiser": c["advertiser"], "account": c["account"], "status": c["status"],
            "reason": c["reason"], "keywords": list(c["keywords"]), "kind": c["kind"], "bid_cents": c["bid_cents"],
            "daily_budget_cents": c["daily_budget_cents"], "spent_today": c["spent_today"],
            "total_spent": c["total_spent"], "impressions": c["impressions"], "landing": dict(c["landing"])}
