"""Operations accounting and problem discovery (module `ops`, host `ops`).

This module is part of the open-world substrate. It makes the *consequences* of everything an organisation does
measurable from authoritative state, without knowing what the organisation is trying to do:

* **Daily P&L journal** (`ops_days`): every ledger posting is attributed to the owner of the account and the day,
  by ledger kind (sale, compute, contract, subscription, energy, freight, ...). It is maintained by an observer of
  the bank's own applier, so it is derived state that replays exactly. `operating_result` excludes financing and
  capital flows (endowments, owner transfers, loans, equity, liquidation, hardware/machines).
* **Loss journal**: world mechanics record *avoidable losses* where they physically happen - requests a service
  could not serve because it was overloaded or down, sales lost to empty shelves, machine downtime, late
  deliveries, failed calls to a dependency. A loss record has a mechanism label (descriptive: "what happened"),
  an estimated value and the resource it happened on. No loss kind names a product or a solution.
* **Problem requests** (`ops.request`): any organisation can publish a buyer request describing a problem in its
  own words, with evidence from its journal, the resources it is willing to delegate and a budget. Requests are
  public and searchable; anybody can respond. This is how demand for things nobody has built yet becomes visible.

Nothing here decides whether a solution is good. Adoption decisions are made by the organisations themselves
(see `crucible.actors.minds`) from measured outcomes in this journal (see `caps` evidence).
"""
from __future__ import annotations

from ..core import schema as S
from ..core.errors import Conflict, Forbidden, InvalidAction, NotFound
from ..core.kernel import SYSTEM, Ctx, Module, applier, endpoint, process, register_module
from ..services.identity import ident

DAY = 86400
# ledger kinds that are not part of operating performance (financing, owner capital, capital expenditure)
NON_OPERATING = frozenset({"endowment", "transfer", "liquidation", "distribution", "income", "loan", "loan_repayment",
                           "equity", "capital", "hardware", "machine", "facility", "dividend", "investment",
                           "savings", "escrow_release"})
SYSTEM_OWNERS = frozenset({SYSTEM, "households", ""})


def operating_result(day_rec: dict | None) -> int:
    if not day_rec:
        return 0
    return int(sum(v for k, v in day_rec["flows"].items() if k not in NON_OPERATING))


@register_module
class Ops(Module):
    name = "ops"
    depends = ("identity", "bank")

    def install(self, world):
        st = world.state
        st.declare_table("ops_days", index=("owner",))
        st.declare_table("ops_requests", index=("by", "status"))
        st.declare_table("ops_responses", index=("request", "by"))
        world.hooks.setdefault("txn_observers", []).append(self._on_txn)
        world.hooks["ops"] = self

    def genesis(self, ctx):
        ctx.schedule(ctx.now + DAY + 600, "ops.daily", {}, prio=80, period=DAY, key="ops.daily")

    # ------------------------------------------------------------------ derived journal (applier side)
    def _rec(self, owner: str, day: int) -> dict:
        r = self.world.state.tables["ops_days"].get(f"{owner}|{day}")
        if r is None:
            return {"owner": owner, "day": day, "flows": {}, "losses": {}, "loss_n": {}}
        return {"owner": owner, "day": day, "flows": dict(r["flows"]), "losses": dict(r["losses"]),
                "loss_n": dict(r["loss_n"])}

    def _on_txn(self, ev) -> None:
        st = self.world.state
        day = ev.t // DAY
        kind = ev.payload["kind"]
        per_owner: dict[str, int] = {}
        for acct, d in ev.payload["postings"]:
            a = st.tables["accounts"].get(acct)
            if a is None or a["kind"] == "system" or a["owner"] in SYSTEM_OWNERS:
                continue
            per_owner[a["owner"]] = per_owner.get(a["owner"], 0) + d
        for owner, d in sorted(per_owner.items()):
            if d == 0:
                continue
            r = self._rec(owner, day)
            r["flows"][kind] = r["flows"].get(kind, 0) + d
            st.put("ops_days", f"{owner}|{day}", r)

    @applier("ops.loss")
    def ap_loss(self, ev):
        p = ev.payload
        r = self._rec(p["owner"], ev.t // DAY)
        r["losses"][p["kind"]] = r["losses"].get(p["kind"], 0) + int(p["cents"])
        r["loss_n"][p["kind"]] = r["loss_n"].get(p["kind"], 0) + int(p.get("n", 1))
        self.world.state.put("ops_days", f"{p['owner']}|{ev.t // DAY}", r)

    # ------------------------------------------------------------------ API used by world mechanics
    def record_loss(self, ctx: Ctx, owner: str, kind: str, cents: int, n: int = 1, resource: str | None = None) -> None:
        """Called by mechanisms where an avoidable loss physically happens (no strategy is implied)."""
        if not owner or owner in SYSTEM_OWNERS or cents <= 0 and n <= 0:
            return
        ctx.emit("ops.loss", {"owner": owner, "kind": kind, "cents": int(max(0, cents)), "n": int(n),
                              "resource": resource}, vis=(owner,))

    def day(self, st, owner: str, day: int) -> dict | None:
        return st.get("ops_days", f"{owner}|{day}")

    def series(self, st, owner: str, days: list[int]) -> list[int]:
        return [operating_result(self.day(st, owner, d)) for d in days]

    def window(self, st, owner: str, n_days: int, today: int) -> dict:
        """Aggregate the last `n_days` complete days: flows by kind, losses by kind, operating result per day."""
        flows: dict[str, int] = {}
        losses: dict[str, int] = {}
        loss_n: dict[str, int] = {}
        per_day = []
        for d in range(today - n_days, today):
            r = self.day(st, owner, d)
            per_day.append(operating_result(r))
            if r:
                for k, v in r["flows"].items():
                    flows[k] = flows.get(k, 0) + v
                for k, v in r["losses"].items():
                    losses[k] = losses.get(k, 0) + v
                for k, v in r["loss_n"].items():
                    loss_n[k] = loss_n.get(k, 0) + v
        revenue = sum(v for k, v in flows.items() if v > 0 and k not in NON_OPERATING)
        return {"flows": flows, "losses": losses, "loss_n": loss_n, "operating_per_day": per_day,
                "revenue": revenue, "operating": sum(per_day)}

    # ------------------------------------------------------------------ endpoints: analytics
    @endpoint("kpis", S.obj({"org": S.s_str("organisation (default: you)", maxLength=40),
                             "days": S.s_int(minimum=1, maximum=90)}), read_only=True)
    def ep_kpis(self, ctx, org=None, days=7):
        """Daily operating journal of you or an organisation you may read (`analytics.read`): ledger flows by kind,
        operating result (excluding financing and capital flows) and recorded losses by mechanism."""
        owner = ident(ctx.world).resolve_handle(ctx.state, org) if org else ctx.actor
        ident(ctx.world).authorize(ctx, owner, "analytics.read")
        today = ctx.now // DAY
        rows = []
        for d in range(today - days + 1, today + 1):
            r = self.day(ctx.state, owner, d)
            rows.append({"day": d, "operating_result": operating_result(r), "flows": dict(r["flows"]) if r else {},
                         "losses": dict(r["losses"]) if r else {}, "loss_events": dict(r["loss_n"]) if r else {}})
        return {"owner": owner, "days": rows}

    # ------------------------------------------------------------------ endpoints: problem requests (buyer side)
    @endpoint("request", S.obj({
        "org": S.s_str("requesting organisation (default: you)", maxLength=40),
        "title": S.s_str(maxLength=160), "body": S.s_str("the problem in your own words", maxLength=4000),
        "tags": S.s_arr(S.s_str(maxLength=40), maxItems=12),
        "evidence_days": S.s_int("attach your loss journal for the last N days", minimum=0, maximum=60),
        "resources": S.s_arr(S.s_str(maxLength=60), "resource ids a solution may operate on", maxItems=12),
        "offered_caps": S.s_arr(S.s_str(maxLength=40), "capabilities you are willing to delegate", maxItems=12),
        "budget_cents_month": S.s_int(minimum=0, maximum=100_000_000),
        "expires_in_days": S.s_int(minimum=1, maximum=180)}, ["title", "body"]), cost_s=60)
    def ep_request(self, ctx, title, body, org=None, tags=None, evidence_days=0, resources=None, offered_caps=None,
                   budget_cents_month=0, expires_in_days=30):
        """Publish a public request for solutions to a problem you have. Anyone can read and respond."""
        st = ctx.state
        owner = ident(ctx.world).resolve_handle(st, org) if org else ctx.actor
        ident(ctx.world).authorize(ctx, owner, "org.admin")
        if len([r for r in st.find("ops_requests", "by", owner) if r["status"] == "open"]) >= 10:
            raise Conflict("at most 10 open requests")
        evidence = None
        if evidence_days:
            w = self.window(st, owner, int(evidence_days), ctx.now // DAY)
            evidence = {"days": int(evidence_days), "losses": w["losses"], "loss_events": w["loss_n"],
                        "revenue": w["revenue"], "verified": True}
        rid = st.new_id("req")
        rec = {"id": rid, "by": owner, "title": title, "body": body, "tags": sorted(set(tags or [])),
               "evidence": evidence, "resources": list(resources or []), "offered_caps": sorted(set(offered_caps or [])),
               "budget_cents_month": int(budget_cents_month), "expires": ctx.now + int(expires_in_days) * DAY}
        ctx.emit("ops.request", rec, vis="public")
        from ..services.search import index
        index(ctx, rid, "request", title, f"{body}\nTags: {' '.join(rec['tags'])}\nLosses: "
              f"{' '.join(sorted((evidence or {}).get('losses', {})))}", owner, ref=rid)
        return {"request": rid}

    @endpoint("requests", S.obj({"status": S.s_enum(["open", "closed", "any"]), "by": S.s_str(maxLength=40),
                                 "limit": S.s_int(minimum=1, maximum=100)}), read_only=True)
    def ep_requests(self, ctx, status="open", by=None, limit=30):
        """Public problem requests (newest first)."""
        st = ctx.state
        ids = st.find_ids("ops_requests", "by", ident(ctx.world).resolve_handle(st, by)) if by else st.ids("ops_requests")
        out = []
        for rid in reversed(ids):
            r = st.tables["ops_requests"][rid]
            if status != "any" and r["status"] != status:
                continue
            out.append({"id": rid, **{k: r[k] for k in ("by", "title", "body", "tags", "evidence", "resources",
                                             "offered_caps", "budget_cents_month", "status", "t", "expires")},
                        "responses": len(st.find_ids("ops_responses", "request", rid))})
            if len(out) >= limit:
                break
        return out

    @endpoint("respond", S.obj({"request": S.s_str(maxLength=40), "text": S.s_str(maxLength=4000),
                                "capability": S.s_str("a published capability id that addresses it", maxLength=40),
                                "price_cents_month": S.s_int(minimum=0, maximum=100_000_000)},
                               ["request", "text"]), cost_s=30)
    def ep_respond(self, ctx, request, text, capability=None, price_cents_month=None):
        """Answer a request (visible to the requester and you)."""
        st = ctx.state
        r = st.get("ops_requests", request)
        if r is None or r["status"] != "open":
            raise NotFound("no open request " + request)
        if capability is not None and not st.has("capabilities", capability):
            raise NotFound("no capability " + capability)
        if len([x for x in st.find("ops_responses", "request", request) if x["by"] == ctx.actor]) >= 3:
            raise Conflict("at most 3 responses per request")
        sid = st.new_id("resp")
        ctx.emit("ops.response", {"id": sid, "request": request, "by": ctx.actor, "text": text,
                                  "capability": capability, "price_cents_month": price_cents_month},
                 vis=(r["by"], ctx.actor))
        return {"response": sid}

    @endpoint("responses", S.obj({"request": S.s_str(maxLength=40)}, ["request"]), read_only=True)
    def ep_responses(self, ctx, request):
        """Responses to a request you made (or your own responses)."""
        st = ctx.state
        r = st.get("ops_requests", request)
        if r is None:
            raise NotFound("no request")
        mine = ident(ctx.world).allowed(ctx, ctx.actor, r["by"], "org.admin")
        return [dict(x) for x in st.find("ops_responses", "request", request) if mine or x["by"] == ctx.actor]

    @endpoint("close_request", S.obj({"request": S.s_str(maxLength=40)}, ["request"]))
    def ep_close(self, ctx, request):
        """Close a request you made."""
        r = ctx.state.get("ops_requests", request)
        if r is None:
            raise NotFound("no request")
        ident(ctx.world).authorize(ctx, r["by"], "org.admin")
        if r["status"] == "open":
            ctx.emit("ops.request_status", {"id": request, "status": "closed"}, vis="public")
        return {"ok": True}

    @process("daily")
    def pr_daily(self, ctx):
        for rid in ctx.state.find_ids("ops_requests", "status", "open"):
            if ctx.state.tables["ops_requests"][rid]["expires"] <= ctx.now:
                ctx.emit("ops.request_status", {"id": rid, "status": "closed"}, vis="public")

    @applier("ops.request")
    def ap_request(self, ev):
        p = dict(ev.payload)
        p.update(status="open", t=ev.t)
        self.world.state.insert("ops_requests", p.pop("id"), p)

    @applier("ops.request_status")
    def ap_request_status(self, ev):
        self.world.state.patch("ops_requests", ev.payload["id"], status=ev.payload["status"])

    @applier("ops.response")
    def ap_response(self, ev):
        p = dict(ev.payload)
        p["t"] = ev.t
        self.world.state.insert("ops_responses", p.pop("id"), p)

    def observe(self, world, actor):
        st = world.state
        n = len([1 for r in st.find("ops_requests", "status", "open")])
        return {"open_requests": n} if n else None


def ops_of(world) -> Ops | None:
    return world.modules.get("ops")  # type: ignore[return-value]
