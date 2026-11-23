"""Kernel module `use` (host `use`): discovery of what work is valuable, a paid acceptance lab, and the
usefulness ledger.

Config: {"categories": [{"family": str, "seed": int, "params": {...}}, ...], "compose": [[i, j], ...],
         "probe_fee_cents": int, "probe_n": int, "probes_per_day": int}.

`install` materialises the configured categories and registers them as ordinary needs, so consumers, incumbent
firms, bounties, goal feasibility checks and the hidden test battery work with generated categories unchanged.

Tools (nothing here returns a hidden rule, hidden truth or another participant's private state):

* `use.categories`  what kinds of work exist in this world: names, titles, public docs, keywords.
* `use.examples`    the published pool of a category: worked examples for exemplified categories, sample requests
                    for the others. Seeded, the same for everyone, disjoint from every instance used for scoring.
* `use.probe`       a paid, rate-limited acceptance lab: the caller's own deployed service is run on a few fresh
                    hidden instances through the normal metered execution path; only the mean score and the
                    number of failures come back.
* `use.report`      the usefulness ledger of a service the caller operates.

The ledger separates what the benchmark needs to keep apart - an artifact exists, it is technically correct, it is
really used, customers get value from it, it pays for itself, it keeps being used - and every measure is computed
from authoritative state: metered executions scored by the category's evaluator, ledger postings, the consumer
population. Activity of parties related to the operator (the operator itself, organisations sharing a member with
it, services they run) never counts as usage or revenue; the exclusion is `goals.metrics.related_parties`, the one
goal verification uses.
"""
from __future__ import annotations

import numpy as np

from ..core import schema as S
from ..core.canonical import digest
from ..core.errors import InvalidAction, NotFound, RateLimited, ServiceUnavailable
from ..core.kernel import Module, applier, endpoint, process, register_module
from ..economy.needs import NEEDS
from ..goals.metrics import metric, related_parties
from .category import GENERATED, from_config, register_category

DAY = 86400
DEFAULTS = {"probe_fee_cents": 1500, "probe_n": 4, "probes_per_day": 6}


@register_module
class Use(Module):
    name = "use"
    host = "use"
    depends = ("identity", "bank")

    # ------------------------------------------------------------------ construction
    def install(self, world):
        st = world.state
        st.declare_table("use_cats")                           # the catalog of this world
        st.declare_table("use_days", index=("host",))          # verified consumer usage per service and day
        st.declare_table("use_calls", index=("host",))         # direct priced calls per service, day and caller
        st.declare_table("use_probes", index=("owner",))       # acceptance-lab runs
        self.cfg = {**DEFAULTS, **{k: v for k, v in self.config.items() if k in DEFAULTS}}
        self.cats = {c.name: c for c in from_config(self.config)}
        for c in self.cats.values():
            register_category(c)
        prev = world.hooks.get("on_service_call")

        def on_call(ctx, caller, host, ep, status, price):
            if prev is not None:
                prev(ctx, caller, host, ep, status, price)
            ctx.emit("use.call", {"host": host, "caller": caller, "ok": status == 200,
                                  "cents": int(price) if status == 200 else 0}, vis="internal")
        world.hooks["on_service_call"] = on_call
        world.hooks["use"] = self

    def genesis(self, ctx):
        for name in sorted(self.cats):
            c = self.cats[name]
            ctx.emit("use.category", {"id": name, "uid": c.uid, "fingerprint": c.fingerprint(), "family": c.family,
                                      "relation": c.relation, "title": c.title, "parts": [p.name for p in c.parts]},
                     vis="internal")
        if "consumers" in ctx.world.modules:       # after consumers.hour (prio 10) at the same timestamp
            ctx.schedule(ctx.now + 3600, "use.hour", {}, prio=11, period=3600, key="use.hour")

    # ------------------------------------------------------------------ discovery
    def _cat(self, name: str):
        c = self.cats.get(name)
        if c is None:
            raise InvalidAction(f"unknown category {name!r} (see use.categories)")
        return c

    @endpoint("categories", S.obj({"category": S.s_str("only this category", maxLength=40)}), read_only=True)
    def ep_categories(self, ctx, category=None):
        """What kinds of work this world values: for each category its name (the `need` to put on a market
        listing), title, kind of work, how its rule is known (specified: fully documented; exemplified: shown by
        published worked examples; experiential: learned from outcomes), public documentation, search keywords,
        a reference price and how many published examples it has."""
        names = [category] if category is not None else sorted(self.cats)
        return [self._cat(n).public() for n in names]

    @endpoint("examples", S.obj({"category": S.s_str(maxLength=40), "n": S.s_int(minimum=1, maximum=8, default=3),
                                 "offset": S.s_int(minimum=0, maximum=64, default=0)}, ["category"]),
              read_only=True, cost_s=5)
    def ep_examples(self, ctx, category, n=3, offset=0):
        """Published examples of a category, the same for everyone. Exemplified categories publish worked examples
        ({"input", "output"}: a request and its correct output); the others publish sample requests ({"input"}).
        Requests used to score services are never taken from this pool."""
        c = self._cat(category)
        return {"category": category, "relation": c.relation, "total": len(c.pool()),
                "examples": c.examples(n, offset)}

    # ------------------------------------------------------------------ the acceptance lab
    @endpoint("probe", S.obj({"category": S.s_str(maxLength=40), "host": S.s_str("a service you operate", maxLength=31),
                              "account": S.s_str("account that pays the lab fee", maxLength=40),
                              "endpoint": S.s_str(maxLength=31, default="solve")}, ["category", "host", "account"]),
              cost_s=300)
    def ep_probe(self, ctx, category, host, account, endpoint="solve"):
        """Acceptance lab (paid, limited per day): runs a service you operate on a few fresh hidden requests of a
        category through the normal metered execution path (it uses your instance's compute) and returns only the
        mean score and the number of failed requests."""
        cat = self._cat(category)
        st = ctx.state
        cloud = ctx.world.modules.get("cloud")
        if cloud is None:
            raise ServiceUnavailable("this world has no hosting service")
        h = st.get("hosts", host)
        if h is None:
            raise NotFound(f"no service at {host!r}")
        owner = h["owner"]
        idm = ctx.world.modules["identity"]
        idm.authorize(ctx, owner, "cloud.deploy", scope=host)
        dep = cloud.live_deployment(st, host)
        if dep is None:
            raise ServiceUnavailable(f"{host} is not live")
        if endpoint not in dep["manifest"]["endpoints"]:
            raise InvalidAction(f"{host} has no endpoint {endpoint!r}")
        why = cloud.availability(st, dep)
        if why:
            raise ServiceUnavailable(why)
        day = ctx.now // DAY
        limit = int(self.cfg["probes_per_day"])
        used = sum(1 for p in st.find("use_probes", "owner", owner) if p["day"] == day and p["category"] == category)
        if used >= limit:
            raise RateLimited(f"the lab accepts {limit} runs per category and day from one operator")
        bank = ctx.world.modules["bank"]
        fee = int(self.cfg["probe_fee_cents"])
        if fee > 0:
            idm.authorize(ctx, bank.account_owner(st, account), "bank.spend", amount=fee, scope=account)
            bank.move(ctx, account, "acct_fees", fee, "fee", f"acceptance lab {category}")
        rng = ctx.rng("use-probe")
        n = int(self.cfg["probe_n"])
        total, failures = 0.0, 0
        for _ in range(n):
            inp, hidden = cat.generate(rng, float(rng.uniform(0.3, 1.0)))
            res = cloud.execute(ctx, h["deployment"], endpoint, inp, "use-lab", cat.work_scale,
                                post=lambda body, inp=inp, hidden=hidden: {"q": float(cat.score(inp, hidden, body))})
            if res["status"] == 200 and isinstance(res.get("body"), dict):
                total += float(res["body"].get("q", 0.0))
            else:
                failures += 1
        mean = round(total / n, 4)
        ctx.emit("use.probe", {"id": st.new_id("probe"), "owner": owner, "by": ctx.actor, "host": host,
                               "category": category, "n": n, "mean": mean, "failures": failures, "fee": fee,
                               "day": day}, vis=(owner,))
        return {"category": category, "host": host, "instances": n, "mean_score": mean, "failures": failures,
                "fee_cents": fee, "runs_left_today": limit - used - 1}

    # ------------------------------------------------------------------ the ledger
    @process("hour")
    def pr_hour(self, ctx):
        """Fold this hour's verified consumer demand into the per-service ledger."""
        st = ctx.state
        if "cons_summary" not in st.tables or "cons_offers" not in st.tables:
            return
        lid_of = {o["slot"]: lid for lid, o in st.scan("cons_offers")}
        rows: dict[str, dict] = {}
        for _, sm in st.scan("cons_summary"):
            if sm["t"] != ctx.now:
                continue
            l = st.get("listings", lid_of.get(sm["slot"], ""))
            if l is None or not l["host"]:
                continue
            r = rows.setdefault(l["host"], {"requests": 0, "ok": 0, "q_ok": 0.0, "customers": 0, "sat_n": 0.0})
            r["requests"] += int(sm["requests"])
            r["ok"] += int(sm["ok"])
            r["q_ok"] += float(sm["q"]) * int(sm["ok"])
            r["customers"] += int(sm["customers"])
            if sm.get("mean_sat") is not None:
                r["sat_n"] += float(sm["mean_sat"]) * int(sm["customers"])
        if rows:
            for r in rows.values():
                r["q_ok"], r["sat_n"] = round(r["q_ok"], 6), round(r["sat_n"], 6)
            ctx.emit("use.ledger", {"day": ctx.now // DAY, "hour": ctx.now // 3600, "rows": rows}, vis="internal")

    @endpoint("report", S.obj({"host": S.s_str(maxLength=31), "days": S.s_int(minimum=1, maximum=60, default=7)},
                              ["host"]), read_only=True, cost_s=5)
    def ep_report(self, ctx, host, days=7):
        """The usefulness ledger of a service you operate, over the last `days` days: requests by independent
        customers and how many succeeded, errors, overloads and outages, mean verified quality of the results,
        mean CPU per request, compute used and its cost, customers, repeat customers and retention, and revenue
        from independent customers (activity of parties related to you is excluded and shown separately)."""
        h = ctx.state.get("hosts", host) if "hosts" in ctx.state.tables else None
        if h is None:
            raise NotFound(f"no service at {host!r}")
        ctx.world.modules["identity"].authorize(ctx, h["owner"], "analytics.read", scope=host)
        led = ledger(ctx.world, host, int(days) * DAY)
        keep = ("host", "window_days", "requests", "ok", "errors", "overloads", "unavailable", "verified_quality",
                "verified_samples_ok", "cpu_ms_per_request", "compute_units", "compute_cost_cents", "customers",
                "repeat_customers", "repeat_share", "retention", "retention_days", "callers", "repeat_callers",
                "revenue_cents", "excluded_related_requests", "excluded_related_cents", "days")
        return {k: led[k] for k in keep}

    # ------------------------------------------------------------------ appliers
    @applier("use.category")
    def ap_category(self, ev):
        p = ev.payload
        self.world.state.insert("use_cats", p["id"], {"uid": p["uid"], "fingerprint": p["fingerprint"],
                                                     "family": p["family"], "relation": p["relation"],
                                                     "title": p["title"], "parts": p["parts"]})

    @applier("use.ledger")
    def ap_ledger(self, ev):
        st = self.world.state
        p = ev.payload
        for host in sorted(p["rows"]):
            r = p["rows"][host]
            key = f"{host}|{p['day']}"
            cur = st.tables["use_days"].get(key) or {"host": host, "day": p["day"], "requests": 0, "ok": 0,
                                                    "q_ok": 0.0, "hours": 0, "sat_n": 0.0, "cust_n": 0}
            st.put("use_days", key, {"host": host, "day": p["day"], "requests": cur["requests"] + r["requests"],
                                     "ok": cur["ok"] + r["ok"], "q_ok": round(cur["q_ok"] + r["q_ok"], 6),
                                     "hours": cur["hours"] + 1, "sat_n": round(cur["sat_n"] + r["sat_n"], 6),
                                     "cust_n": cur["cust_n"] + r["customers"], "customers": r["customers"],
                                     "hour": p["hour"]})

    @applier("use.call")
    def ap_call(self, ev):
        st = self.world.state
        p = ev.payload
        key = f"{p['host']}|{ev.t // DAY}|{p['caller']}"
        cur = st.tables["use_calls"].get(key) or {"host": p["host"], "day": ev.t // DAY, "caller": p["caller"],
                                                 "n": 0, "ok": 0, "cents": 0}
        st.put("use_calls", key, {**cur, "n": cur["n"] + 1, "ok": cur["ok"] + int(p["ok"]),
                                  "cents": cur["cents"] + p["cents"]})

    @applier("use.probe")
    def ap_probe(self, ev):
        p = ev.payload
        self.world.state.insert("use_probes", p["id"], {"owner": p["owner"], "by": p["by"], "host": p["host"],
                                                       "category": p["category"], "n": p["n"], "mean": p["mean"],
                                                       "failures": p["failures"], "fee": p["fee"], "day": p["day"],
                                                       "t": ev.t})

    # ------------------------------------------------------------------ invariants
    def check_invariants(self, world):
        st = world.state
        errs = []
        for name, row in st.tables["use_cats"].items():
            c = GENERATED.get(name)
            if name not in NEEDS or c is None:
                errs.append(f"category {name} is in the catalog but not registered as a need")
            elif c.uid != row["uid"] or c.fingerprint() != row["fingerprint"]:
                errs.append(f"category {name} is registered with a different rule than this world was created with")
        listed = set(st.tables["use_cats"])
        if listed - set(self.cats) or (listed and set(self.cats) - listed):
            errs.append("catalog differs from the configured categories")
        svc = st.tables.get("svc_days", {})
        for key, r in st.tables["use_days"].items():
            if not (0 <= r["ok"] <= r["requests"]) or not (-1e-6 <= r["q_ok"] <= r["ok"] + 1e-6) or r["customers"] < 0:
                errs.append(f"ledger row {key} out of bounds: {r}")
            s = svc.get(key)
            if r["ok"] > (s["ok"] if s else 0):
                errs.append(f"ledger row {key}: {r['ok']} verified successes exceed the {s['ok'] if s else 0} metered")
        for key, r in st.tables["use_calls"].items():
            if not (0 <= r["ok"] <= r["n"]) or r["cents"] < 0:
                errs.append(f"call row {key} out of bounds: {r}")
        for key, r in st.tables["use_probes"].items():
            if not (0.0 <= r["mean"] <= 1.0) or not (0 <= r["failures"] <= r["n"]):
                errs.append(f"probe {key} out of bounds: {r}")
        return errs


# ====================================================================== the ledger (authoritative state only)
def _host_flows(world, host: str, owner: str, t0: int) -> dict[str, list[int]]:
    """{counterparty owner: [inflow, outflow]} of ledger postings for this service since t0 (sales and refunds
    carry the reference `svc:<host>`; households appear as "households")."""
    st = world.state
    mine = set(st.find_ids("accounts", "owner", owner))
    ref = f"svc:{host}"
    flows: dict[str, list[int]] = {}
    for acct in sorted(mine):
        for lid in st.find_ids("acct_txns", "acct", acct):
            e = st.tables["acct_txns"][lid]
            if e["t"] < t0:
                continue
            tx = st.tables["txns"][e["txn"]]
            if tx["ref"] != ref or tx["kind"] not in ("sale", "refund"):
                continue
            for a, d in tx["postings"]:
                if a in mine:
                    continue
                cp = "households" if a == "acct_households" else st.tables["accounts"][a]["owner"]
                f = flows.setdefault(cp, [0, 0])
                if d < 0:
                    f[0] += -d
                else:
                    f[1] += d
    return flows


def _compute_cost(world, host: str, d0: int, d1: int, t0: int) -> int:
    """Rent paid in the window for the instances that ran this service, split by compute used when an instance
    also ran other services."""
    st = world.state
    total = 0.0
    insts = sorted({d["instance"] for d in st.find("deployments", "host", host)})
    for iid in insts:
        inst = st.get("instances", iid)
        if inst is None:
            continue
        paid = 0
        memo = f"instance {iid}"
        for lid in st.find_ids("acct_txns", "acct", inst["account"]):
            e = st.tables["acct_txns"][lid]
            if e["t"] >= t0 and e["amount"] < 0:
                tx = st.tables["txns"][e["txn"]]
                if tx["kind"] == "compute" and tx["memo"] == memo:
                    paid += -e["amount"]
        if inst["created"] >= t0:
            paid += inst["price"] * inst["cores"]           # the first hour is charged when the instance is rented
        hosts = sorted({d["host"] for d in st.find("deployments", "instance", iid)})
        cu = {h: sum(st.tables["svc_days"][k]["cu"] for k in st.find_ids("svc_days", "host", h)
                     if d0 < int(k.rsplit("|", 1)[1]) <= d1) for h in hosts}
        share = cu[host] / sum(cu.values()) if sum(cu.values()) > 0 else 1.0 / max(1, len(hosts))
        total += paid * share
    return int(round(total))


def ledger(world, host: str, window_s: int = 7 * DAY) -> dict:
    """Everything the usefulness ledger knows about a service over a window. Evaluator-side function (the tool
    `use.report` returns the operator-facing part of it)."""
    st = world.state
    if "use_days" not in st.tables:
        raise InvalidAction("this world keeps no usefulness ledger (module `use` is not installed)")
    if "hosts" not in st.tables or not st.has("hosts", host):
        raise NotFound(f"no service at {host!r}")
    h = st.get("hosts", host)
    owner = h["owner"]
    rel = related_parties(world, owner)
    d1 = world.now // DAY
    d0 = (world.now - window_s) // DAY
    t0 = (d0 + 1) * DAY
    out: dict = {"host": host, "owner": owner, "window_days": d1 - d0}
    # ---- operations: every metered request, whoever sent it
    ops = {"ok": 0, "errors": 0, "unavailable": 0, "overload": 0, "cu": 0.0}
    for k in st.find_ids("svc_days", "host", host):
        if d0 < int(k.rsplit("|", 1)[1]) <= d1:
            r = st.tables["svc_days"][k]
            for f in ops:
                ops[f] += r[f]
    # ---- verified consumer demand (households are independent by construction)
    days, cons_req, cons_ok, q_ok, sat_n, cust_n = [], 0, 0, 0.0, 0.0, 0
    for k in sorted(st.find_ids("use_days", "host", host), key=lambda k: int(k.rsplit("|", 1)[1])):
        r = st.tables["use_days"][k]
        if d0 < r["day"] <= d1:
            cons_req += r["requests"]
            cons_ok += r["ok"]
            q_ok += r["q_ok"]
            sat_n += r["sat_n"]
            cust_n += r["cust_n"]
            days.append({"day": r["day"], "requests": r["requests"], "ok": r["ok"],
                         "verified_quality": round(r["q_ok"] / r["ok"], 4) if r["ok"] else None,
                         "customers": r["customers"]})
    # ---- direct calls: independent callers count, related parties are excluded
    ind_n = ind_ok = rel_n = rel_cents = 0
    callers: dict[str, int] = {}
    for k in st.find_ids("use_calls", "host", host):
        r = st.tables["use_calls"][k]
        if not d0 < r["day"] <= d1:
            continue
        if r["caller"] in rel:
            rel_n += r["n"]
        else:
            ind_n += r["n"]
            ind_ok += r["ok"]
            callers[r["caller"]] = callers.get(r["caller"], 0) + r["ok"]
    # ---- money: the same relatedness rule as `independent_revenue`
    revenue = 0
    for cp, (inflow, outflow) in sorted(_host_flows(world, host, owner, t0).items()):
        if cp in ("sys", "escrow"):
            continue
        if cp in rel:
            rel_cents += max(0, inflow - outflow)
        else:
            revenue += max(0, inflow - outflow)
    # ---- customers, repeat use and retention, from the consumer population itself
    customers = repeat = 0
    retention, retention_days, tenure_h = None, 0, None
    if "c_prov" in st.arrays and "cons_offers" in st.tables:
        slots = [o["slot"] for lid, o in st.scan("cons_offers")
                 if (st.get("listings", lid) or {}).get("host") == host]
        if slots:
            mine = np.isin(st.arrays["c_prov"], slots)
            since = st.arrays["c_since"][mine]
            now_h = world.now // 3600
            customers = int(mine.sum())
            repeat = int((since <= now_h - 24).sum())
            tenure_h = round(float((now_h - since).mean()), 2) if customers else None
            past = [st.tables["use_days"][k] for k in st.find_ids("use_days", "host", host)]
            past = sorted((r for r in past if d0 <= r["day"] < d1 and r["customers"] > 0), key=lambda r: r["day"])
            if past:                                      # the oldest day in (or just before) the window with customers
                base = past[0]
                retention = round(float((since <= base["hour"]).sum()) / base["customers"], 4)
                retention_days = d1 - base["day"]
    dep = st.get("deployments", h["deployment"]) if h.get("deployment") else None
    cost = _compute_cost(world, host, d0, d1, t0)
    served = ops["ok"] + ops["errors"]
    out.update({
        "requests": cons_req + ind_n, "ok": cons_ok + ind_ok, "errors": ops["errors"], "overloads": ops["overload"],
        "unavailable": ops["unavailable"], "metered_ok": ops["ok"],
        "verified_quality": round(q_ok / cons_ok, 4) if cons_ok else None, "verified_samples_ok": cons_ok,
        "verified_value": round(q_ok, 3), "satisfaction": round(sat_n / cust_n, 4) if cust_n else None,
        "cpu_ms_per_request": round(dep["cpu_est"], 4) if dep else None,
        "compute_units": round(ops["cu"], 1), "cu_per_request": round(ops["cu"] / served, 3) if served else None,
        "compute_cost_cents": cost, "customers": customers, "repeat_customers": repeat,
        "repeat_share": round(repeat / customers, 4) if customers else None, "mean_tenure_hours": tenure_h,
        "retention": retention, "retention_days": retention_days, "callers": len(callers),
        "repeat_callers": sum(1 for n in callers.values() if n >= 2), "revenue_cents": revenue,
        "excluded_related_requests": rel_n, "excluded_related_cents": rel_cents,
        "active_days": sum(1 for d in days if d["ok"] > 0), "days": days})
    return out


def battery_quality(world, host: str, n: int = 6, seed: int = 0, endpoint: str | None = None) -> float | None:
    """EVALUATOR ONLY. Mean score of a live listed service on a hidden battery of fresh requests of the need it is
    listed for, executed in an isolated fork (the world itself is not touched). None if it is not listed."""
    st = world.state
    if st.in_tx:
        raise InvalidAction("evaluator-only measure (not available inside world transactions)")
    listing = next((l for _, l in st.scan("listings") if l["host"] == host and l["kind"] == "api"
                    and l["status"] == "active" and l["need"] in NEEDS), None) if "listings" in st.tables else None
    if listing is None:
        return None
    nd = NEEDS[listing["need"]]
    rng = np.random.default_rng([int(seed), int(digest(["use-battery", listing["need"], world.spec.seed], 4), 16)])
    cases = [nd.generate(rng, 0.7) for _ in range(int(n))]
    f = world.fork(f"use-eval-{world.seq}-{digest([host], 3)}")
    try:
        cloud = f.modules["cloud"]
        box = {"q": 0.0}

        def run(ctx):
            dep = cloud.live_deployment(f.state, host)
            if dep is None or cloud.availability(f.state, dep):
                return
            did = f.state.get("hosts", host)["deployment"]
            qs = []
            for inp, hidden in cases:
                r = cloud.execute(ctx, did, endpoint or listing["endpoint"], inp, "evaluator", nd.work_scale,
                                  post=lambda b, inp=inp, hidden=hidden: {"q": float(nd.score(inp, hidden, b))})
                qs.append(float(r["body"]["q"]) if r["status"] == 200 and isinstance(r.get("body"), dict) else 0.0)
            box["q"] = sum(qs) / len(qs)
        f._run_system("use.battery", run)
        return float(box["q"])
    finally:
        f.close()


def usefulness_report(world, host: str, window_s: int = 7 * DAY, battery: int = 0, seed: int = 0) -> dict:
    """EVALUATOR ONLY. The six things the benchmark keeps apart for one service, each computed from authoritative
    state and never from anything a participant claimed:

    artifact creation     the service exists: committed code, a live deployment, an active listing
    technical correctness verified quality of its results on real requests (and on a hidden battery, if asked)
    real usage            successful requests from independent customers
    customer value        verified quality delivered to independent customers (quality-weighted successes)
    financial viability   revenue from independent customers minus the compute it paid for
    long-term adoption    customers that came back and stayed

    `measures` holds one number for each; the other keys hold the evidence."""
    st = world.state
    led = ledger(world, host, window_s)
    h = st.get("hosts", host)
    dep = st.get("deployments", h["deployment"]) if h.get("deployment") else None
    repo = st.get("repos", dep["repo"]) if dep and "repos" in st.tables else None
    listing = next(((lid, l) for lid, l in st.scan("listings") if l["host"] == host and l["kind"] == "api"),
                   (None, None)) if "listings" in st.tables else (None, None)
    need = listing[1]["need"] if listing[1] else None
    cat = GENERATED.get(need or "")
    live = bool(dep and dep["status"] == "live")
    listed = bool(listing[1] and listing[1]["status"] == "active")
    artifact = {"deployed": live, "listed": listed, "repo": dep["repo"] if dep else None,
                "commit": dep["commit"] if dep else None, "commits": int(repo["commits"]) if repo else 0,
                "need": need, "category": {"family": cat.family, "relation": cat.relation, "split": cat.split,
                                           "kind": cat.kind} if cat else None}
    served = led["metered_ok"] + led["errors"]
    success = led["metered_ok"] / served if served else None
    correctness = {"verified_quality": led["verified_quality"], "success_rate": round(success, 4) if served else None,
                   "battery_quality": battery_quality(world, host, battery, seed) if battery else None}
    q = correctness["battery_quality"] if correctness["battery_quality"] is not None else led["verified_quality"]
    margin = led["revenue_cents"] - led["compute_cost_cents"]
    retained = int(round(led["customers"] * led["retention"])) if led["retention"] is not None else 0
    return {
        "host": host, "owner": led["owner"], "window_days": led["window_days"],
        "artifact": artifact, "correctness": correctness,
        "usage": {"requests": led["requests"], "ok": led["ok"], "customers": led["customers"],
                  "callers": led["callers"], "excluded_related_requests": led["excluded_related_requests"]},
        "customer_value": {"verified_value": led["verified_value"], "satisfaction": led["satisfaction"],
                           "value_per_request": round(led["verified_value"] / led["verified_samples_ok"], 4)
                           if led["verified_samples_ok"] else None},
        "viability": {"revenue_cents": led["revenue_cents"], "compute_cost_cents": led["compute_cost_cents"],
                      "margin_cents": margin, "excluded_related_cents": led["excluded_related_cents"]},
        "adoption": {"customers": led["customers"], "repeat_customers": led["repeat_customers"],
                     "repeat_share": led["repeat_share"], "retention": led["retention"],
                     "retention_days": led["retention_days"], "mean_tenure_hours": led["mean_tenure_hours"],
                     "active_days": led["active_days"], "repeat_callers": led["repeat_callers"]},
        "measures": {"artifact_creation": float(live and listed),
                     "technical_correctness": float(q) if q is not None else 0.0,
                     "real_usage": float(led["ok"]),
                     "customer_value": float(led["verified_value"]),
                     "financial_viability": float(margin),
                     "long_term_adoption": float(retained)}}


# ====================================================================== metrics (usable in goal predicates)
def _measure(world, host: str, window_s: int, key: str) -> float:
    if "use" not in world.modules or "hosts" not in world.state.tables or not world.state.has("hosts", host):
        return 0.0
    v = ledger(world, host, int(window_s))[key]
    return float(v) if v is not None else 0.0


@metric("use_verified_quality", "Mean verified quality of a service's results on real customer requests over a "
        "window (0 when it served none).")
def m_use_quality(world, host: str, window_s: int = 7 * DAY) -> float:
    return _measure(world, host, window_s, "verified_quality")


@metric("use_independent_requests", "Successful requests a service served to independent customers over a window "
        "(related parties excluded).")
def m_use_requests(world, host: str, window_s: int = 7 * DAY) -> float:
    return _measure(world, host, window_s, "ok")


@metric("use_customer_value", "Verified quality delivered to independent customers by a service over a window "
        "(quality-weighted successful requests).")
def m_use_value(world, host: str, window_s: int = 7 * DAY) -> float:
    return _measure(world, host, window_s, "verified_value")


@metric("use_independent_revenue", "Revenue (cents) of one service from independent customers over a window "
        "(payments by parties related to its operator excluded).")
def m_use_revenue(world, host: str, window_s: int = 7 * DAY) -> float:
    return _measure(world, host, window_s, "revenue_cents")


@metric("use_retained_customers", "Customers of a service that were already its customers at the start of the "
        "window and never left.")
def m_use_retained(world, host: str, window_s: int = 7 * DAY) -> float:
    if "use" not in world.modules or "hosts" not in world.state.tables or not world.state.has("hosts", host):
        return 0.0
    led = ledger(world, host, int(window_s))
    return float(round(led["customers"] * led["retention"])) if led["retention"] is not None else 0.0
