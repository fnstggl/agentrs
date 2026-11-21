"""Scripted inventor agents: competent-agent *controls* for the open-world substrate.

Like `Builder`, these encode engineering knowledge a capable developer would have; they use only public tools and
public information and are not known to any world mechanism. They exist to show that the world admits invention
of new kinds of software that independent organisations then evaluate and adopt (or reject) on measured value -
not as evidence of learning.

* `CapacityInventor`  reads public problem requests, picks ones whose verified evidence shows overload losses,
                      writes an hourly forecasting autoscaler (a new service kind: it operates *other firms'*
                      infrastructure through delegated `cloud.manage`), deploys it, publishes it as a capability and
                      answers the requests. Variant `algorithm="naive"` ships a deliberately poor controller.
* `PeakPricingInventor` addresses the same pain with a different mechanism: time-of-day pricing of the firm's
                      listing (delegated `market.manage`), no extra compute.
* `MaintenanceInventor` (production worlds) predicts machine failures from noisy telemetry and schedules
                      maintenance (delegated `make.maintain`).
"""
from __future__ import annotations

import json

from .scripted import ScriptPolicy, call, ok

AUTOSCALER = '''import math
import crucible_sdk as sdk


def _demand(r):
    if not r:
        return 0.0
    served = r["ok"] + r["errors"]
    if served:
        return r["cu"] * (served + r["overload"]) / served
    return 0.0


def tick(args, caller):
    """Size the instance for the next hourly demand batch: same hour yesterday (and the day before), scaled by
    recent growth, with a safety margin. Overloaded hours are extrapolated from the served/turned-away ratio."""
    host = args["host"]
    m = sdk.call("cloud.metrics", host=host, hours=50)
    inst = m["instance"]                      # wherever the service currently runs
    hrs = {r["hour"]: r for r in m["hours"]}
    now_h = max(hrs)
    cur = max(1, m["cores"])
    per_core = m["capacity_cu_per_hour"] / cur
    target = now_h + 1
    y1, y2 = _demand(hrs.get(target - 24)), _demand(hrs.get(target - 48))
    last = sum(_demand(hrs.get(h)) for h in range(now_h - 23, now_h + 1))
    prev = sum(_demand(hrs.get(h)) for h in range(now_h - 47, now_h - 23))
    growth = min(1.5, max(0.8, last / prev)) if prev > 0 else 1.0
    base = max(y1, 0.9 * y2)
    fc = base * growth * MARGIN
    need = max(1, min(32, int(math.ceil(fc / per_core)))) if fc > 0 else cur
    if need != m["cores"]:
        sdk.call("cloud.resize", instance=inst, cores=need)
    return {"cores": need, "forecast_cu": round(fc, 1)}
'''

NAIVE = '''import crucible_sdk as sdk


def tick(args, caller):
    """Naive controller: always scale down to one core unless the last hour overloaded."""
    m = sdk.call("cloud.metrics", host=args["host"], hours=2)
    last = m["hours"][-1]
    need = m["cores"] + 1 if last["overload"] > 0 else 1
    if need != m["cores"]:
        sdk.call("cloud.resize", instance=args["instance"], cores=need)
    return {"cores": need}
'''

PEAK_PRICING = '''import crucible_sdk as sdk

BASE = {}


def tick(args, caller):
    """Time-of-day pricing: price above the daily mean in hours that overloaded yesterday, below it at night."""
    lid, host = args["listing"], args["host"]
    m = sdk.call("cloud.metrics", host=host, hours=26)
    hrs = {r["hour"]: r for r in m["hours"]}
    now_h = max(hrs)
    y = hrs.get(now_h + 1 - 24) or {}
    l = sdk.call("market.view", listing=lid)
    base = BASE.setdefault(lid, l["price_cents"])
    if y.get("overload", 0) > 0:
        p = int(base * 1.4) + 1
    elif y.get("cu", 0) < 0.25 * m["capacity_cu_per_hour"]:
        p = max(1, int(base * 0.8))
    else:
        p = base
    if p != l["price_cents"]:
        sdk.call("market.update", listing=lid, price_cents=p)
    return {"price": p}
'''


def manifest(endpoints: dict, description: str) -> str:
    return json.dumps({"entry": "service.py", "description": description, "endpoints": endpoints}, indent=1)


class _AppMixin:
    def deploy_app(self, org, acct, name, files, host, cores=1):
        r = yield call("code.create_repo", name=name, owner=org)
        if not ok(r):
            return None
        repo = r["result"]["repo"]
        r = yield call("code.commit", repo=repo, files=files, message="initial version")
        if not ok(r):
            return None
        r = yield call("cloud.providers")
        provs = sorted((p for p in (r or {}).get("result", []) if p["cores_free"] >= cores and not p["in_outage"]),
                       key=lambda p: (p["price_core_hour_cents"] * (1 + p["outages_30d"] * 0.05), p["provider"]))
        if not provs:
            return None
        r = yield call("cloud.rent", provider=provs[0]["provider"], cores=cores, account=acct, owner=org)
        if not ok(r):
            return None
        inst = r["result"]["instance"]
        r = yield call("cloud.deploy", repo=repo, host=host, instance=inst)
        if not ok(r) or r["result"].get("status") != "live":
            return None
        return {"repo": repo, "instance": inst, "host": host}

    def find_requests(self, loss_kinds, min_events=1):
        r = yield call("ops.requests", status="open", limit=100)
        out = []
        for q in (r or {}).get("result", []) if ok(r) else []:
            ev = q.get("evidence") or {}
            n = sum(ev.get("loss_events", {}).get(k, 0) for k in loss_kinds)
            if n >= min_events or any(k in q.get("tags", []) for k in loss_kinds):
                out.append(q)
        return out

    def answer(self, requests, cap, text):
        answered = getattr(self, "answered", set())
        for q in requests:
            if q["id"] in answered:
                continue
            yield call("ops.respond", request=q["id"], text=text, capability=cap)
            answered.add(q["id"])
        self.answered = answered


class CapacityInventor(_AppMixin, ScriptPolicy):
    name = "capacity_inventor"
    kind = "scripted"

    def __init__(self, capital=300_000, price_per_call=30, algorithm="forecast", margin=1.25, host="autoscale",
                 open_source=False, wait_for_demand=True):
        self.capital, self.price, self.algorithm, self.margin = capital, price_per_call, algorithm, margin
        self.host, self.open_source, self.wait_for_demand = host, open_source, wait_for_demand

    def script(self):
        reqs = []
        while self.wait_for_demand and not reqs:        # invent only once someone has the problem
            reqs = yield from self.find_requests(["capacity_overload"])
            if not reqs:
                yield from self.wait(6)
        org, acct, _ = yield from self.found_company("scale", self.capital)
        if acct is None:
            return
        src = AUTOSCALER.replace("MARGIN", repr(float(self.margin))) if self.algorithm == "forecast" else NAIVE
        files = {"crucible.json": manifest({"tick": {"handler": "tick", "price_cents": int(self.price),
                                                     "doc": "Resize the bound instance for the next hour.",
                                                     "params": {"type": "object", "additionalProperties": True}}},
                                           "Hourly forecasting autoscaler for cloud services"),
                 "service.py": src}
        app = yield from self.deploy_app(org, acct, "autoscaler", files, self.host)
        if app is None:
            return
        r = yield call("caps.publish", host=self.host, title="Autoscaler: right-size your instance every hour",
                       description="Your service's demand changes during the day. This capability reads your "
                                   "service's hourly metrics and resizes its instance before each hour so peaks are "
                                   "served and idle cores are released at night. Needs cloud.manage and "
                                   "analytics.read on the service.",
                       tags=["capacity", "overload", "scaling", "instance", "demand", "peak"],
                       params={"host": {"kind": "host"}},
                       requires=[{"cap": "cloud.manage", "scope": "$host"},
                                 {"cap": "analytics.read", "scope": "$host"}],
                       triggers=[{"endpoint": "tick", "every_s": 3600, "args": {"host": "$host"}}],
                       open_source=self.open_source)
        if not ok(r):
            return
        self.capability = r["result"]["capability"]
        text = (f"Capability {self.capability} sizes your instance hour by hour from your own metrics. "
                f"{self.price} cents per hourly run; install it as a trial and judge it on your own books.")
        yield from self.answer(reqs, self.capability, text)
        while True:
            yield from self.wait(24)
            reqs = yield from self.find_requests(["capacity_overload"])
            yield from self.answer(reqs, self.capability, text)


class PeakPricingInventor(_AppMixin, ScriptPolicy):
    name = "peak_pricing_inventor"
    kind = "scripted"

    def __init__(self, capital=300_000, price_per_call=20, host="peakprice"):
        self.capital, self.price, self.host = capital, price_per_call, host

    def script(self):
        reqs = []
        while not reqs:
            reqs = yield from self.find_requests(["capacity_overload"])
            if not reqs:
                yield from self.wait(6)
        org, acct, _ = yield from self.found_company("price", self.capital)
        if acct is None:
            return
        files = {"crucible.json": manifest({"tick": {"handler": "tick", "price_cents": int(self.price),
                                                     "doc": "Set this hour's price for the bound listing.",
                                                     "params": {"type": "object", "additionalProperties": True}}},
                                           "Time-of-day pricing"), "service.py": PEAK_PRICING}
        app = yield from self.deploy_app(org, acct, "peakprice", files, self.host)
        if app is None:
            return
        r = yield call("caps.publish", host=self.host, title="Peak pricing for overloaded services",
                       description="Charges more in the hours that overload your service and less at night, so "
                                   "demand moves to when you have capacity. Needs market.manage on your listing and "
                                   "analytics.read on your host.",
                       tags=["pricing", "overload", "peak", "demand", "listing"],
                       params={"host": {"kind": "host"}, "listing": {"kind": "listing"}},
                       requires=[{"cap": "market.manage", "scope": "$listing"},
                                 {"cap": "analytics.read", "scope": "$host"}],
                       triggers=[{"endpoint": "tick", "every_s": 3600,
                                  "args": {"host": "$host", "listing": "$listing"}}])
        if not ok(r):
            return
        self.capability = r["result"]["capability"]
        text = f"Capability {self.capability}: time-of-day pricing that moves demand off your peak hours."
        yield from self.answer(reqs, self.capability, text)
        while True:
            yield from self.wait(24)
            reqs = yield from self.find_requests(["capacity_overload"])
            yield from self.answer(reqs, self.capability, text)


MAINTENANCE = """import crucible_sdk as sdk

THRESH = THRESH_VALUE


def tick(args, caller):
    \"\"\"Predictive maintenance: repair broken machines; maintain idle machines whose vibration (a noisy wear
    signal) is above a threshold.\"\"\"
    fac = args["facility"]
    now = sdk.call("world.clock")["time"]
    acted = []
    for m in sdk.call("make.machines", facility=fac):
        if m["status"] == "broken":
            sdk.call("make.repair", machine=m["machine"])
            acted.append(["repair", m["machine"]])
            continue
        if m["status"] != "ok" or m["busy_until"] > now:
            continue
        t = sdk.call("make.telemetry", machine=m["machine"])
        if t["vibration"] > THRESH and t["hours_since_maintenance"] > 8:
            sdk.call("make.maintain", machine=m["machine"])
            acted.append(["maintain", m["machine"]])
    return {"acted": acted}
"""


class MaintenanceInventor(_AppMixin, ScriptPolicy):
    """Reads requests whose evidence shows machine downtime or defects and offers predictive maintenance."""
    name = "maintenance_inventor"
    kind = "scripted"

    def __init__(self, capital=300_000, price_per_call=15, threshold=0.32, host="predmaint"):
        self.capital, self.price, self.threshold, self.host = capital, price_per_call, threshold, host

    def script(self):
        reqs = []
        while not reqs:
            reqs = yield from self.find_requests(["machine_downtime", "defects"])
            if not reqs:
                yield from self.wait(6)
        org, acct, _ = yield from self.found_company("maint", self.capital)
        if acct is None:
            return
        files = {"crucible.json": manifest({"tick": {"handler": "tick", "price_cents": int(self.price),
                                                     "doc": "Check the facility's machines; maintain worn ones.",
                                                     "params": {"type": "object", "additionalProperties": True}}},
                                           "Predictive maintenance"),
                 "service.py": MAINTENANCE.replace("THRESH_VALUE", repr(float(self.threshold)))}
        app = yield from self.deploy_app(org, acct, "predmaint", files, self.host)
        if app is None:
            return
        r = yield call("caps.publish", host=self.host, title="Predictive maintenance from machine telemetry",
                       description="Watches the vibration telemetry of every machine in your facility every hour, "
                                   "maintains machines before they break down and repairs broken ones at once. "
                                   "Fewer breakdowns, less machine downtime, fewer defects. Needs make.read and "
                                   "make.maintain on the facility.",
                       tags=["maintenance", "machine", "downtime", "breakdown", "failure", "predictive", "defects"],
                       params={"facility": {"kind": "facility"}},
                       requires=[{"cap": "make.read", "scope": "$facility"},
                                 {"cap": "make.maintain", "scope": "$facility"}],
                       triggers=[{"endpoint": "tick", "every_s": 3600, "args": {"facility": "$facility"}}])
        if not ok(r):
            return
        self.capability = r["result"]["capability"]
        text = f"Capability {self.capability}: telemetry-driven maintenance for your machines. Trial it on your books."
        yield from self.answer(reqs, self.capability, text)
        while True:
            yield from self.wait(24)
            reqs = yield from self.find_requests(["machine_downtime", "defects"])
            yield from self.answer(reqs, self.capability, text)


class ProcessResearcher(_AppMixin, ScriptPolicy):
    """Industrial R&D control: when producers publicly report defects, breakdowns, energy costs or curtailment, it
    founds a lab, hires operators, buys machines of the models it wants to understand, runs a designed experiment
    over the (speed, temperature) space on cheap inputs, keeps the operating point with the lowest energy per good
    unit among high-yield points, and sells that knowledge (`make.publish_process`). Experiments cost money, energy,
    labour and time, and measurements are noisy."""
    name = "process_researcher"
    kind = "scripted"

    def __init__(self, models=("form-A", "form-B"), capital=4_000_000, price=60_000, region="r1", grid=(0.2, 0.5, 0.8),
                 units=12):
        self.models, self.capital, self.price, self.region = list(models), capital, price, region
        self.grid, self.units = list(grid), units
        self.findings = {}

    def script(self):
        reqs = []
        while not reqs:
            reqs = yield from self.find_requests(["defects", "machine_downtime", "energy_curtailment"])
            reqs += [q for q in ((yield call("ops.requests", status="open", limit=100)) or {}).get("result", [])
                     if "cost-energy" in q.get("tags", [])]
            if not reqs:
                yield from self.wait(6)
        org, acct, _ = yield from self.found_company("lab", self.capital)
        if acct is None:
            return
        r = yield call("make.build_facility", org=org, region=self.region, name="R&D lab")
        if not ok(r):
            return
        fac = r["result"]["facility"]
        # staff: post a job and hire the first applicants
        yield call("jobs.post", org=org, title="Process technician", role="operator", salary_cents_month=260_000,
                   description="Run experiments on our machines.")
        hired = 0
        for _ in range(72):
            r = yield call("jobs.openings", role="operator", limit=50)
            mine = [p for p in (r or {}).get("result", []) if p.get("org") == org] if ok(r) else []
            for p in mine:
                ra = yield call("jobs.applications", posting=p["posting"])
                for a in (ra or {}).get("result", []) if ok(ra) else []:
                    if a.get("status") == "applied" and hired < 2:
                        rr = yield call("jobs.offer", application=f"{p['posting']}|{a['applicant']}",
                                        salary_cents_month=260_000)
                        hired += 1 if ok(rr) else 0
            if hired >= 2:
                break
            yield None
        machines = {}
        for model in self.models:
            r = yield call("make.buy_machine", facility=fac, model=model)
            if ok(r):
                machines[model] = r["result"]["due"]
        # inputs: the cheapest material on offer
        r = yield call("make.offers", stage="material")
        offers = sorted((o for o in (r or {}).get("result", []) if o["available"] > 0), key=lambda o: o["price_cents"])
        if not offers:
            return
        mat = offers[0]["sku"]
        need = self.units * len(self.grid) ** 2 * len(self.models) + 40
        got = 0
        for o in offers:
            if o["sku"] != mat or got >= need:
                continue
            q = min(o["available"], need - got)
            rr = yield call("make.order", offer=o["offer"], qty=q, to_facility=fac)
            got += q if ok(rr) else 0
        while True:                                 # wait for deliveries
            r = yield call("make.machines", org=org)
            if ok(r) and len(r["result"]) >= len(machines):
                break
            yield from self.wait(6)
        mach_by_model = {m["model"]: m["machine"] for m in r["result"]}
        r = yield call("make.design", org=org, name="probe", op="form", inputs=[{"sku": mat, "qty": 1}], form="sheet")
        if not ok(r):
            return
        probe = r["result"]["design"]
        for model, mid in mach_by_model.items():
            results = []
            for sp in self.grid:
                for tp in self.grid:
                    while True:
                        rr = yield call("make.run", machine=mid, design=probe, units=self.units, speed=sp, temp=tp)
                        if ok(rr):
                            break
                        yield from self.wait(2)
                    run_id = rr["result"]["run"]
                    while True:
                        yield from self.wait(1)
                        r2 = yield call("make.runs", org=org, limit=60)
                        run = next((x for x in (r2 or {}).get("result", []) if x["run"] == run_id), None) if ok(r2) else None
                        if run and run["status"] != "running":
                            break
                    y = run["made"] / max(1, run["units"])
                    e = run["kwh"] / max(1, run["made"]) if run["made"] else 99.0
                    results.append((sp, tp, y, e))
            top = max(r_[2] for r_ in results)
            good = [r_ for r_ in results if r_[2] >= 0.9 * top]
            sp, tp, y, e = min(good, key=lambda r_: r_[3])
            self.findings[model] = {"speed": sp, "temp": tp, "yield": y, "kwh_per_good_unit": e, "grid": results}
            yield call("make.publish_process", org=org, model=model, speed=sp, temp=tp, price_cents=self.price,
                       note=f"Operating point for {model} from a {len(results)}-point designed experiment: "
                            f"higher yield and less energy per good unit than common settings.")
        while True:
            yield from self.wait(24)


PLACEMENT = """import math
import crucible_sdk as sdk


def _demand(r):
    if not r:
        return 0.0
    served = r["ok"] + r["errors"]
    return r["cu"] * (served + r["overload"]) / served if served else 0.0


def tick(args, caller):
    \"\"\"Placement: when the service will need more cores next hour than its provider has free, move it to the
    cheapest provider that can host the forecast (rent there, redeploy, release the old instance); then size it.\"\"\"
    host = args["host"]
    m = sdk.call("cloud.metrics", host=host, hours=50)
    hrs = {r["hour"]: r for r in m["hours"]}
    now_h = max(hrs)
    cur = max(1, m["cores"])
    per_core = m["capacity_cu_per_hour"] / cur
    fc = max(_demand(hrs.get(now_h + 1 - 24)), 0.9 * _demand(hrs.get(now_h + 1 - 48))) * MARGIN
    need = max(1, min(32, int(math.ceil(fc / per_core)))) if fc > 0 else cur
    provs = {p["provider"]: p for p in sdk.call("cloud.providers")}
    here = provs.get(m["provider"])
    moved = None
    if need > cur and here is not None and here["cores_free"] < need - cur:
        options = sorted((p["price_core_hour_cents"], name) for name, p in provs.items()
                         if name != m["provider"] and p["cores_free"] >= need and not p["in_outage"])
        if options and options[0][0] <= MAX_PREMIUM * m["price_core_hour_cents"]:
            new = sdk.call("cloud.rent", provider=options[0][1], cores=need, account=args["account"],
                           owner=m["owner"])["instance"]
            sdk.call("cloud.deploy", repo=m["repo"], host=host, instance=new)
            sdk.call("cloud.release", instance=m["instance"])
            moved = options[0][1]
            return {"moved_to": moved, "cores": need}
    if need != cur:
        try:
            sdk.call("cloud.resize", instance=m["instance"], cores=need)
        except sdk.WorldError:
            return {"cores": cur, "resize_failed": True}
    return {"cores": need, "moved_to": moved}
"""


class PlacementBroker(_AppMixin, ScriptPolicy):
    """Second-order control: answers reports that a provider ran out of capacity, with a service that sizes a
    customer's deployment and moves it to another provider when its own is full at the moment it needs to grow.
    It needs more delegation than an autoscaler (deploy rights on the service, spending on one account)."""
    name = "placement_broker"
    kind = "scripted"

    def __init__(self, capital=300_000, price_per_call=40, margin=1.15, max_premium=1.6, host="placer",
                 open_source=False):
        self.capital, self.price, self.margin, self.max_premium = capital, price_per_call, margin, max_premium
        self.host, self.open_source = host, open_source

    def script(self):
        reqs = []
        while not reqs:
            reqs = yield from self.find_requests(["capacity_unavailable"])
            if not reqs:
                yield from self.wait(6)
        org, acct, _ = yield from self.found_company("place", self.capital)
        if acct is None:
            return
        src = PLACEMENT.replace("MARGIN", repr(float(self.margin))).replace("MAX_PREMIUM", repr(float(self.max_premium)))
        files = {"crucible.json": manifest({"tick": {"handler": "tick", "price_cents": int(self.price),
                                                     "doc": "Size the service and move it if its provider is full.",
                                                     "params": {"type": "object", "additionalProperties": True}}},
                                           "Capacity-aware placement across providers"), "service.py": src}
        app = yield from self.deploy_app(org, acct, "placer", files, self.host)
        if app is None:
            return
        r = yield call("caps.publish", host=self.host,
                       title="Placement broker: never be stuck on a full provider at peak",
                       description="When your provider has no free cores at the hour your service needs to grow, "
                                   "this capability rents capacity at the cheapest provider that has it, moves your "
                                   "service there and releases the old instance; otherwise it sizes your instance "
                                   "hour by hour. Needs cloud.manage, cloud.deploy and analytics.read on the "
                                   "service and bank.spend on the account that pays for compute.",
                       tags=["capacity", "provider", "full", "unavailable", "placement", "migrate", "peak", "cores"],
                       params={"host": {"kind": "host"}, "account": {"kind": "account"}},
                       requires=[{"cap": "cloud.manage", "scope": "$host"}, {"cap": "cloud.deploy", "scope": "$host"},
                                 {"cap": "analytics.read", "scope": "$host"},
                                 {"cap": "bank.spend", "scope": "$account", "limit": 2_000_000}],
                       triggers=[{"endpoint": "tick", "every_s": 3600, "args": {"host": "$host", "account": "$account"}}],
                       open_source=self.open_source)
        if not ok(r):
            return
        self.capability = r["result"]["capability"]
        text = f"Capability {self.capability} moves your service to a provider with free capacity when yours is full."
        yield from self.answer(reqs, self.capability, text)
        while True:
            yield from self.wait(24)
            reqs = yield from self.find_requests(["capacity_unavailable"])
            yield from self.answer(reqs, self.capability, text)


SITE_SRC = r'''
import html
PAGES = PAGES_JSON

def page(req, caller):
    key = req["params"].get("slug", "")
    title, body = PAGES.get(key, PAGES[""])
    links = "".join("<li><a href='/p/%s'>%s</a></li>" % (k, html.escape(v[0])) for k, v in sorted(PAGES.items()) if k)
    return {"html": "<!doctype html><html><head><title>%s</title></head><body><h1>%s</h1><p>%s</p><ul>%s</ul>"
                    "</body></html>" % (html.escape(title), html.escape(title), html.escape(body), links)}
'''


class SiteAuthor(_AppMixin, ScriptPolicy):
    """Competent control for visibility tasks: publishes a small, honest web site about a topic on the synthetic
    internet (a guide page per keyword) and leaves it to the crawler and the search index."""
    name = "site_author"
    kind = "scripted"

    def __init__(self, topic: str, capital=150_000, host=None):
        self.topic, self.capital, self.host = topic, capital, host

    def script(self):
        org, acct, _ = yield from self.found_company("guides", self.capital)
        if acct is None:
            return
        words = [w for w in self.topic.split() if w.isalpha()][:6]
        pages = {"": (f"Guide: {' '.join(words[:3])}", f"A practical guide to {self.topic}. " * 3)}
        for wd in words:
            pages[wd] = (f"{wd.title()} explained", f"How {wd} relates to {self.topic}: causes, measurement, options.")
        files = {"crucible.json": json.dumps({"entry": "service.py", "web": {
            "title": pages[""][0], "routes": {"GET /": "page", "GET /p/{slug}": "page"}}}),
            "service.py": SITE_SRC.replace("PAGES_JSON", repr(pages))}
        host = self.host or f"{self.handle}-guides"[:30]
        app = yield from self.deploy_app(org, acct, "guides", files, host)
        self.site = app
        while True:
            yield from self.wait(24)


DIGEST_SRC = r'''
import html
import crucible_sdk as sdk


def page(req, caller):
    rows = []
    for c in sdk.call("caps.list", limit=50):
        ev = [e for e in sdk.call("caps.evidence", capability=c["capability"]) if e["kind"] == "outcome" and e["public"]]
        for e in ev:
            d = e["data"]
            rows.append("<tr><td>%s</td><td>%s</td><td>%+.0f</td><td>%.0f</td><td>%s</td></tr>" % (
                html.escape(c["title"][:60]), html.escape(e["by"]), float(d.get("effect_per_day") or 0),
                float(d.get("se") or 0), html.escape(e["evidence"])))
    return {"html": "<!doctype html><html><head><title>Verified trial outcomes</title></head><body>"
                    "<h1>Verified trial outcomes of published capabilities</h1><table><tr><th>capability</th>"
                    "<th>firm</th><th>effect (cents/day)</th><th>se</th><th>evidence</th></tr>" + "".join(rows) +
                    "</table></body></html>"}
'''


class EvidenceAggregator(_AppMixin, ScriptPolicy):
    """Information-system control: collects *public, verified* trial outcomes of every capability and distributes
    them - a live web page on the synthetic internet (rendered from the evidence registry at request time) and a
    daily forum digest carrying `#evidence <id>` references that any reader can check. It never invents claims:
    each line points at a registry record."""
    name = "evidence_aggregator"
    kind = "scripted"

    def __init__(self, capital=120_000, host="trialwatch", every_h=24):
        self.capital, self.host, self.every_h = capital, host, every_h

    def script(self):
        org, acct, _ = yield from self.found_company("watch", self.capital)
        if acct is None:
            return
        files = {"crucible.json": json.dumps({"entry": "service.py", "web": {
            "title": "Trial watch", "routes": {"GET /": "page"}}}), "service.py": DIGEST_SRC}
        self.site = yield from self.deploy_app(org, acct, "trialwatch", files, self.host)
        posted: set = set()
        self.posts = 0
        while True:
            r = yield call("caps.list", limit=100)
            new = []
            for c in (r or {}).get("result", []) if ok(r) else []:
                e = yield call("caps.evidence", capability=c["capability"])
                for x in (e or {}).get("result", []) if ok(e) else []:
                    if x["kind"] == "outcome" and x["public"] and x["verified"] and x["evidence"] not in posted:
                        new.append((c, x))
            if new:
                body = "\n".join(f"{c['title'][:50]}: {x['by']} measured {float(x['data'].get('effect_per_day') or 0):+.0f} "
                                 f"cents/day (se {float(x['data'].get('se') or 0):.0f}). #evidence {x['evidence']}"
                                 for c, x in new[:8])
                p = yield call("forum.post", board="general", title=f"Trial watch: {len(new[:8])} verified outcomes",
                               body=body + f"\nAll outcomes: http://{self.host}.crucible/")
                if ok(p):
                    posted.update(x["evidence"] for _, x in new[:8])
                    self.posts += 1
            yield from self.wait(self.every_h)
