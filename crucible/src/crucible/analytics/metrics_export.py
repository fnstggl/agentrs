"""World metrics as a JSON snapshot and in the Prometheus text exposition format (read-only).

`world_metrics(world)` collects, from authoritative state:
  time / seq, principals by kind (and persons by kind: agent / npc / human), money supply by sector (households,
  system accounts, escrow pools, persons, firms, cloud providers, lenders, service principals), transactions per
  day (count, volume, by kind) over a trailing window, services live, consumer requests ok / failed per host over
  the window, customers per host, open support tickets, contracts and claims by status, and the kernel's stats
  (commands, rejected, events, process runs) with an events-per-second rate measured between calls.

`prometheus_text(world)` renders the same numbers as `crucible_*` metric families with labels; serve it from any
HTTP endpoint (the dashboard exposes `/w/<id>/metrics`).
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict

from . import DAY, has_table

_RATE: dict[str, tuple[float, int, int]] = {}   # world id -> (wall time, events, sim time) at the previous call


def _sector(world, acct: dict) -> str:
    st = world.state
    owner = acct["owner"]
    if owner == "sys":
        name = acct.get("name") or ""
        if acct.get("kind") == "escrow" or "escrow" in name:
            return "escrow"
        return {"households": "households", "genesis": "genesis", "external": "external",
                "fees": "fees"}.get(name, "system")
    if owner.startswith("p:"):
        rec = st.get("persons", owner) if has_table(world, "persons") else None
        return f"persons_{rec['kind']}" if rec else "persons"
    if owner.startswith("o:"):
        if has_table(world, "providers") and st.has("providers", owner):
            return "providers"
        if has_table(world, "npc_lenders") and st.has("npc_lenders", owner):
            return "lenders"
        if has_table(world, "npc_firms") and st.has("npc_firms", owner):
            return "incumbent_firms"
        return "firms"
    if owner.startswith("svc:"):
        return "services"
    return "other"


def world_metrics(world, window_s: int = DAY) -> dict:
    """JSON-serialisable snapshot of the world's state metrics (see module docstring)."""
    st = world.state
    now = world.now
    t0 = now - window_s
    m: dict = {"world_id": world.world_id, "family": world.spec.family, "seed": world.spec.seed,
               "spec_hash": world.spec.hash()[:16], "time": now, "day": round(now / DAY, 4), "seq": world.seq,
               "window_s": window_s}
    kinds: dict[str, int] = defaultdict(int)
    status: dict[str, int] = defaultdict(int)
    for _, p in st.scan("principals"):
        kinds[p["kind"]] += 1
        status[p.get("status", "active")] += 1
    m["principals"] = dict(kinds)
    m["principals_by_status"] = dict(status)
    if has_table(world, "persons"):
        pk: dict[str, int] = defaultdict(int)
        for _, p in st.scan("persons"):
            pk[p["kind"]] += 1
        m["persons"] = dict(pk)
    if has_table(world, "orgs"):
        ok: dict[str, int] = defaultdict(int)
        for _, o in st.scan("orgs"):
            ok[o["status"]] += 1
        m["orgs"] = dict(ok)
    # ---- money
    if has_table(world, "accounts"):
        sec: dict[str, int] = defaultdict(int)
        n_acc: dict[str, int] = defaultdict(int)
        total = 0
        for _, a in st.scan("accounts"):
            s = _sector(world, a)
            sec[s] += int(a["balance"])
            n_acc[s] += 1
            total += int(a["balance"])
        m["money_by_sector"] = dict(sorted(sec.items()))
        m["accounts_by_sector"] = dict(sorted(n_acc.items()))
        m["money_total"] = total  # double-entry: always 0
        circ = sum(v for k, v in sec.items() if k not in ("genesis", "external", "fees", "system"))
        m["money_in_circulation"] = circ
    if has_table(world, "txns"):
        n = 0
        vol = 0
        by_kind: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for tid in reversed(st.ids("txns")):
            tx = st.tables["txns"][tid]
            if tx["t"] <= t0:
                break
            amt = sum(d for _, d in tx["postings"] if d > 0)
            n += 1
            vol += amt
            by_kind[tx["kind"]][0] += 1
            by_kind[tx["kind"]][1] += amt
        days = max(window_s / DAY, 1e-9)
        m["txns_window"] = n
        m["txns_per_day"] = round(n / days, 3)
        m["txn_volume_per_day"] = int(vol / days)
        m["txns_by_kind"] = {k: {"n": v[0], "volume": v[1]} for k, v in sorted(by_kind.items())}
        m["txns_total"] = st.count("txns")
    # ---- services and infrastructure
    if has_table(world, "deployments"):
        ds: dict[str, int] = defaultdict(int)
        for _, d in st.scan("deployments"):
            ds[d["status"]] += 1
        m["deployments"] = dict(ds)
        m["services_live"] = ds.get("live", 0)
    if has_table(world, "instances"):
        ist: dict[str, int] = defaultdict(int)
        cores = 0
        for _, i in st.scan("instances"):
            ist[i["status"]] += 1
            if i["status"] == "running":
                cores += i["cores"]
        m["instances"] = dict(ist)
        m["cores_rented"] = cores
    if has_table(world, "providers"):
        m["providers"] = {pid: {"cores": p["cores"], "price": p["price"], "status": p["status"],
                                "in_outage": bool(p["outage_start"] <= now < p["outage_until"])}
                          for pid, p in st.scan("providers")}
    if has_table(world, "svc_days"):
        d0, d1 = (t0 // DAY), (now // DAY)
        req: dict[str, dict[str, int]] = {}
        for k, r in st.scan("svc_days"):
            host, day = k.rsplit("|", 1)
            if not (d0 < int(day) <= d1 or (window_s < DAY and int(day) == d1)):
                continue
            x = req.setdefault(host, {"ok": 0, "errors": 0, "unavailable": 0, "overload": 0, "revenue_cents": 0})
            for f in x:
                x[f] += int(r.get(f, 0))
        m["requests_by_host"] = req
        m["requests_ok"] = sum(v["ok"] for v in req.values())
        m["requests_failed"] = sum(v["errors"] + v["unavailable"] + v["overload"] for v in req.values())
    cons = world.modules.get("consumers")
    if cons is not None and "c_prov" in st.arrays:
        shares = cons.market_shares(world)
        m["customers_by_host"] = dict(sorted(shares.items()))
        prov = st.arrays["c_prov"]
        m["consumers"] = int(len(prov))
        m["consumers_unserved"] = int((prov < 0).sum())
        if has_table(world, "accounts") and st.has("accounts", "acct_households"):
            m["household_cash"] = int(st.tables["accounts"]["acct_households"]["balance"])
    if has_table(world, "sup_tickets"):
        ts: dict[str, int] = defaultdict(int)
        for _, t in st.scan("sup_tickets"):
            ts[t["status"]] += 1
        m["tickets"] = dict(ts)
        m["tickets_open"] = sum(v for k, v in ts.items() if k != "resolved")
    if has_table(world, "contracts"):
        cs: dict[str, int] = defaultdict(int)
        for _, c in st.scan("contracts"):
            cs[c["status"]] += 1
        m["contracts"] = dict(cs)
    if has_table(world, "claims"):
        cl: dict[str, int] = defaultdict(int)
        for _, c in st.scan("claims"):
            cl[c["status"]] += 1
        m["claims"] = dict(cl)
    if has_table(world, "listings"):
        ls: dict[str, int] = defaultdict(int)
        for _, l in st.scan("listings"):
            ls[f"{l['kind']}:{l['status']}"] += 1
        m["listings"] = dict(ls)
    # ---- kernel
    stats = dict(world.stats)
    wall = time.monotonic()
    prev = _RATE.get(world.world_id)
    rate = None
    sim_rate = None
    if prev is not None and wall > prev[0] and world.seq >= prev[1]:
        rate = (world.seq - prev[1]) / (wall - prev[0])
        sim_rate = (world.now - prev[2]) / (wall - prev[0])
    _RATE[world.world_id] = (wall, world.seq, world.now)
    stats["events_per_sec"] = None if rate is None else round(rate, 3)
    stats["sim_seconds_per_sec"] = None if sim_rate is None else round(sim_rate, 3)
    stats["events_per_sim_day"] = round(world.seq / max(now / DAY, 1e-9), 1) if now else None
    m["kernel"] = stats
    return m


# ====================================================================== Prometheus exposition
def _esc(v) -> str:
    return str(v).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _num(v) -> str:
    if v is None:
        return "NaN"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "NaN" if math.isnan(v) else ("+Inf" if v > 0 else "-Inf")
    return repr(float(v)) if isinstance(v, float) else str(int(v))


class _Prom:
    def __init__(self, base: dict):
        self.lines: list[str] = []
        self.base = base
        self.declared: set[str] = set()

    def fam(self, name: str, kind: str, help_: str) -> None:
        if name not in self.declared:
            self.declared.add(name)
            self.lines.append(f"# HELP {name} {help_}")
            self.lines.append(f"# TYPE {name} {kind}")

    def sample(self, name: str, value, **labels) -> None:
        lab = dict(self.base, **labels)
        ls = ",".join(f'{k}="{_esc(v)}"' for k, v in lab.items())
        self.lines.append(f"{name}{{{ls}}} {_num(value)}")


def prometheus_text(world, metrics: dict | None = None) -> str:
    """Prometheus text exposition (format 0.0.4) of `world_metrics(world)`."""
    m = metrics if metrics is not None else world_metrics(world)
    p = _Prom({"world": m["world_id"]})
    p.fam("crucible_world_time_seconds", "gauge", "Simulated world time in seconds.")
    p.sample("crucible_world_time_seconds", m["time"])
    p.fam("crucible_world_seq", "counter", "Sequence number of the last event.")
    p.sample("crucible_world_seq", m["seq"])
    p.fam("crucible_principals", "gauge", "Principals by kind.")
    for k, v in sorted(m.get("principals", {}).items()):
        p.sample("crucible_principals", v, kind=k)
    if "persons" in m:
        p.fam("crucible_persons", "gauge", "Persons by kind (agent, npc, human).")
        for k, v in sorted(m["persons"].items()):
            p.sample("crucible_persons", v, kind=k)
    if "orgs" in m:
        p.fam("crucible_orgs", "gauge", "Organisations by status.")
        for k, v in sorted(m["orgs"].items()):
            p.sample("crucible_orgs", v, status=k)
    if "money_by_sector" in m:
        p.fam("crucible_money_cents", "gauge", "Sum of account balances by sector (cents).")
        for k, v in m["money_by_sector"].items():
            p.sample("crucible_money_cents", v, sector=k)
        p.fam("crucible_money_in_circulation_cents", "gauge", "Money held outside system accounts (cents).")
        p.sample("crucible_money_in_circulation_cents", m["money_in_circulation"])
    if "txns_per_day" in m:
        p.fam("crucible_txns_per_day", "gauge", "Ledger transactions per simulated day over the window.")
        p.sample("crucible_txns_per_day", m["txns_per_day"])
        p.fam("crucible_txn_volume_cents_per_day", "gauge", "Ledger volume per simulated day over the window.")
        p.sample("crucible_txn_volume_cents_per_day", m["txn_volume_per_day"])
        p.fam("crucible_txns_window", "gauge", "Transactions in the window by kind.")
        for k, v in m["txns_by_kind"].items():
            p.sample("crucible_txns_window", v["n"], kind=k)
        p.fam("crucible_txns_total", "counter", "All ledger transactions.")
        p.sample("crucible_txns_total", m["txns_total"])
    if "deployments" in m:
        p.fam("crucible_deployments", "gauge", "Deployments by status.")
        for k, v in sorted(m["deployments"].items()):
            p.sample("crucible_deployments", v, status=k)
        p.fam("crucible_services_live", "gauge", "Live deployed services.")
        p.sample("crucible_services_live", m["services_live"])
    if "instances" in m:
        p.fam("crucible_instances", "gauge", "Compute instances by status.")
        for k, v in sorted(m["instances"].items()):
            p.sample("crucible_instances", v, status=k)
        p.fam("crucible_cores_rented", "gauge", "Cores of running instances.")
        p.sample("crucible_cores_rented", m["cores_rented"])
    if "providers" in m:
        p.fam("crucible_provider_cores", "gauge", "Provider capacity in cores.")
        p.fam("crucible_provider_outage", "gauge", "1 while a provider is in an outage.")
        for k, v in m["providers"].items():
            p.sample("crucible_provider_cores", v["cores"], provider=k)
            p.sample("crucible_provider_outage", v["in_outage"], provider=k)
    if "requests_by_host" in m:
        p.fam("crucible_requests", "gauge", "Service requests over the window by host and outcome.")
        for h, v in sorted(m["requests_by_host"].items()):
            for k in ("ok", "errors", "unavailable", "overload"):
                p.sample("crucible_requests", v[k], host=h, outcome=k)
        p.fam("crucible_service_revenue_cents", "gauge", "Service revenue over the window by host (cents).")
        for h, v in sorted(m["requests_by_host"].items()):
            p.sample("crucible_service_revenue_cents", v["revenue_cents"], host=h)
    if "customers_by_host" in m:
        p.fam("crucible_customers", "gauge", "Simulated consumers currently served by each host.")
        for h, v in m["customers_by_host"].items():
            p.sample("crucible_customers", v, host=h)
        p.fam("crucible_consumers_unserved", "gauge", "Simulated consumers without a provider.")
        p.sample("crucible_consumers_unserved", m["consumers_unserved"])
    if "tickets" in m:
        p.fam("crucible_tickets", "gauge", "Support tickets by status.")
        for k, v in sorted(m["tickets"].items()):
            p.sample("crucible_tickets", v, status=k)
        p.fam("crucible_tickets_open", "gauge", "Unresolved support tickets.")
        p.sample("crucible_tickets_open", m["tickets_open"])
    for fam, key, lab in (("crucible_contracts", "contracts", "status"), ("crucible_claims", "claims", "status"),
                          ("crucible_listings", "listings", "kind_status")):
        if key in m:
            p.fam(fam, "gauge", f"{key.title()} by {lab}.")
            for k, v in sorted(m[key].items()):
                p.sample(fam, v, **{lab: k})
    k = m.get("kernel", {})
    p.fam("crucible_kernel_commands_total", "counter", "Commands executed by this process since load.")
    p.sample("crucible_kernel_commands_total", k.get("commands", 0))
    p.fam("crucible_kernel_rejected_total", "counter", "Commands rejected since load.")
    p.sample("crucible_kernel_rejected_total", k.get("rejected", 0))
    p.fam("crucible_kernel_events_total", "counter", "Events committed by this process since load.")
    p.sample("crucible_kernel_events_total", k.get("events", 0))
    p.fam("crucible_kernel_process_runs_total", "counter", "Scheduled process runs since load.")
    p.sample("crucible_kernel_process_runs_total", k.get("procs", 0))
    p.fam("crucible_kernel_events_per_second", "gauge", "Event throughput (wall clock) between two scrapes.")
    p.sample("crucible_kernel_events_per_second", k.get("events_per_sec"))
    return "\n".join(p.lines) + "\n"


def write_json(world, path: str, **kw) -> dict:
    m = world_metrics(world, **kw)
    with open(path, "w") as f:
        json.dump(m, f, indent=1, sort_keys=True, default=str)
    return m
