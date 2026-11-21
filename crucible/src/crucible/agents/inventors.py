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
    host, inst = args["host"], args["instance"]
    m = sdk.call("cloud.metrics", host=host, hours=50)
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
                                   "served and idle cores are released at night. Needs cloud.manage on the "
                                   "instance and analytics.read on the host.",
                       tags=["capacity", "overload", "scaling", "instance", "demand", "peak"],
                       params={"host": {"kind": "host"}, "instance": {"kind": "instance"}},
                       requires=[{"cap": "cloud.manage", "scope": "$instance"},
                                 {"cap": "analytics.read", "scope": "$host"}],
                       triggers=[{"endpoint": "tick", "every_s": 3600,
                                  "args": {"host": "$host", "instance": "$instance"}}],
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
