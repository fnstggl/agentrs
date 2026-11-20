"""Baseline and control policies.

* `IdlePolicy`      - does nothing (reward-hacking probe: inactivity must not pay).
* `RandomPolicy`    - random tools with random schema-valid arguments (floor; exploit probe).
* `BountyHunter`    - simple heuristic: solves open bounties for need formats it knows.
* `Builder`         - scripted competent entrepreneur: writes a strong implementation itself, starts a company,
                      deploys, lists, advertises, operates (scales, answers tickets, reprices).
* `Employer`        - scripted organiser: never writes the algorithm; hires a simulated developer, assigns the work
                      through the issue tracker, then deploys and sells what the employee committed.
* `Composer`        - scripted integrator: builds a service on top of other firms' APIs (calls them from its own
                      sandboxed code and returns the best answer), paying them per call.

The scripted agents encode *developer and business knowledge* (good algorithms, sensible prices). They use only
public tools and public information; they are scientific controls showing that the economy admits several
distinct legitimate strategies, not privileged actors. Nothing in the world mechanics knows about them.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..economy.needs import NEEDS, best_tier, tier_source
from .base import Policy


def call(tool: str, **args) -> dict:
    return {"tool": tool, "args": args}


def ok(r) -> bool:
    return bool(r) and r.get("ok") and not (isinstance(r.get("result"), dict) and r["result"].get("status") in (402, 500, 503))


class IdlePolicy(Policy):
    name, kind = "idle", "baseline"

    def act(self, obs, tools):
        return None


class RandomPolicy(Policy):
    name, kind = "random", "random"

    def act(self, obs, tools):
        tools = [t for t in tools if not t["name"].startswith("admin.")]
        t = tools[int(self.rng.integers(0, len(tools)))]
        return {"tool": t["name"], "args": _random_args(t["inputSchema"], self.rng, self.actor)}


def _random_args(schema: dict, rng, actor: str) -> dict:
    out = {}
    props = schema.get("properties", {})
    req = set(schema.get("required", []))
    for k, p in props.items():
        if k not in req and rng.random() < 0.6:
            continue
        out[k] = _random_value(p, rng, actor)
    return out


def _random_value(p, rng, actor):
    t = p.get("type")
    if isinstance(t, list):
        t = t[0]
    if "enum" in p:
        return p["enum"][int(rng.integers(0, len(p["enum"])))]
    if t == "integer":
        lo, hi = p.get("minimum", 0), p.get("maximum", 10_000)
        return int(rng.integers(lo, min(hi, lo + 100_000) + 1))
    if t == "number":
        return float(rng.uniform(p.get("minimum", 0.0), p.get("maximum", 100.0)))
    if t == "boolean":
        return bool(rng.random() < 0.5)
    if t == "array":
        return [_random_value(p.get("items", {"type": "string"}), rng, actor) for _ in range(int(rng.integers(0, 3)))]
    if t == "object":
        return {}
    choices = [actor, actor.split(":")[-1], "acct_000001", "hello", "o:standards", "x" * int(rng.integers(1, 12))]
    return str(choices[int(rng.integers(0, len(choices)))])[:p.get("maxLength", 40)]


class ScriptPolicy(Policy):
    """Write a script as a generator: `r = yield call(...)` gives the ActionResult dict; `yield None` waits."""
    kind = "scripted"

    def reset(self, actor, rng):
        super().reset(actor, rng)
        self.handle = actor.split(":", 1)[1]
        self.gen = self.script()
        self.started = False
        self.last = None
        self.finished = False
        self.log: list[str] = []

    def act(self, obs, tools):
        self.obs = obs
        if self.finished:
            return None
        try:
            if not self.started:
                self.started = True
                a = next(self.gen)
            else:
                a = self.gen.send(self.last)
        except StopIteration:
            self.finished = True
            return None
        self.last = None
        return a

    def observe_result(self, action, result):
        self.last = result
        if not result.get("ok"):
            self.log.append(f"{action['tool']}: {result.get('error', {}).get('code')}: {result.get('error', {}).get('message', '')[:120]}")

    def script(self):
        yield None

    # ------------------------------------------------------------------ reusable steps
    def wait(self, n):
        for _ in range(n):
            yield None

    def personal_account(self):
        r = yield call("bank.accounts")
        mine = [a for a in (r or {}).get("result", []) if a["owner"] == self.actor]
        return mine[0]["account"] if mine else None

    def found_company(self, suffix: str, capital: int):
        personal = yield from self.personal_account()
        org_handle = f"{self.handle}-{suffix}"[:31]
        r = yield call("id.create_org", handle=org_handle, name=f"{self.handle.title()} {suffix.title()}", kind="company")
        if not ok(r):
            return None, None, personal
        org = r["result"]["org"]
        r = yield call("biz.register", org=org, account=personal, sector=suffix)
        r = yield call("bank.open_account", owner=org, name="operating")
        if not ok(r):
            return org, None, personal
        acct = r["result"]["account"]
        yield call("bank.transfer", from_account=personal, to=acct, amount=int(capital), memo="founder capital")
        return org, acct, personal

    def launch_service(self, org, acct, repo, host, need, price, entry="service.py"):
        r = yield call("cloud.providers")
        provs = sorted((p for p in (r or {}).get("result", []) if p["cores_free"] >= 1 and not p["in_outage"]),
                       key=lambda p: (p["price_core_hour_cents"] * (1 + p["outages_30d"] * 0.05), p["provider"]))
        if not provs:
            return None
        r = yield call("cloud.rent", provider=provs[0]["provider"], cores=1, account=acct, owner=org)
        if not ok(r):
            return None
        inst = r["result"]["instance"]
        r = yield call("cloud.deploy", repo=repo, host=host, instance=inst)
        if not ok(r) or r["result"].get("status") != "live":
            return None
        nd = NEEDS[need]
        r = yield call("market.list", kind="api", title=f"{host}: {nd.title}", price_cents=int(price), seller=org,
                       host=host, endpoint="solve", need=need,
                       description=f"{nd.title}. Fast and correct. Keywords: {' '.join(nd.keywords)} {host}.crucible")
        listing = r["result"]["listing"] if ok(r) else None
        return {"instance": inst, "listing": listing, "host": host}

    def operate(self, org, acct, svc, repo, need, every=6):
        """Maintenance loop: scale on overload, answer tickets, reprice against competition."""
        host = svc["host"]
        while True:
            r = yield call("cloud.status", host=host, days=2)
            if ok(r):
                days = r["result"]["days"]
                over = sum(d.get("overload", 0) for d in days[-1:])
                full = days[-2] if len(days) > 1 else (days[-1] if days else {})
                rev_day = full.get("revenue_cents", 0)
                r0 = yield call("cloud.instances")
                cur0 = [i for i in (r0 or {}).get("result", []) if i["instance"] == svc["instance"]]
                cores0 = cur0[0]["cores"] if cur0 else 1
                price_core_h = cur0[0]["price_core_hour_cents"] if cur0 else 200
                # scale only if a day's revenue would still cover the bigger instance with margin; else reprice
                affordable = rev_day > 1.5 * (cores0 + 1) * price_core_h * 24
                if over > 0 and not affordable and svc.get("listing"):
                    r5 = yield call("market.view", listing=svc["listing"])
                    if ok(r5):
                        yield call("market.update", listing=svc["listing"],
                                   price_cents=int(r5["result"]["price_cents"] * 1.15) + 1)
                if over > 0 and affordable:
                    r2 = yield call("cloud.instances")
                    cur = [i for i in (r2 or {}).get("result", []) if i["instance"] == svc["instance"]]
                    cores = (cur[0]["cores"] if cur else 1) + 1
                    prov = cur[0]["provider"] if cur else None
                    if prov:
                        r3 = yield call("cloud.rent", provider=prov, cores=cores, account=acct, owner=org)
                        if ok(r3):
                            new = r3["result"]["instance"]
                            r4 = yield call("cloud.deploy", repo=repo, host=host, instance=new)
                            if ok(r4) and r4["result"].get("status") == "live":
                                yield call("cloud.release", instance=svc["instance"])
                                svc["instance"] = new
            r = yield call("support.list", owner=org, status="unresolved", limit=10)
            tickets = r["result"] if ok(r) and isinstance(r["result"], list) else (r or {}).get("result", {}).get("tickets", []) if ok(r) else []
            for t in tickets[:5]:
                tid = t.get("ticket") or t.get("id")
                if tid:
                    yield call("support.resolve", ticket=tid, note="Thanks, we scaled up and fixed the issue.")
            yield from self.wait(every)


def service_files(need: str, source: str, price: int, entry: str = "service.py") -> dict:
    nd = NEEDS[need]
    manifest = {"entry": entry, "description": nd.title,
                "endpoints": {"solve": {"handler": "solve", "doc": nd.doc[:300], "price_cents": int(price),
                                        "params": {"type": "object", "additionalProperties": True}}}}
    files = {"crucible.json": json.dumps(manifest)}
    if source is not None:
        files[entry] = source
    return files


def cheapest_rival(rivals, need, factor=0.9):
    prices = [r["price_cents"] for r in rivals if r.get("need") == need]
    base = min(prices) if prices else int(NEEDS[need].base_price_cents * 1.5)
    return max(1, int(base * factor))


class Builder(ScriptPolicy):
    name = "builder"

    def __init__(self, need: str | None = None, capital: int = 400_000, undercut: float = 0.9, advertise: bool = True,
                 license_price: int | None = None):
        self.need, self.capital, self.undercut, self.advertise = need, capital, undercut, advertise
        self.license_price = license_price

    def choose_need(self):
        r = yield call("market.browse", kind="api", sort="rating", limit=30)
        rows = r["result"] if ok(r) else []
        best: dict[str, float] = {}
        for x in rows:
            if x.get("need"):
                best[x["need"]] = max(best.get(x["need"], 0.0), x.get("rating") or 3.0)
        cands = sorted(best, key=lambda n: (best[n], n))      # weakest incumbents first
        return cands[0] if cands else "sort.records"

    def script(self):
        need = self.need or (yield from self.choose_need())
        r = yield call("market.browse", need=need, sort="price", limit=10)
        price = cheapest_rival(r["result"] if ok(r) else [], need, self.undercut)
        org, acct, _ = yield from self.found_company("labs", self.capital)
        if acct is None:
            return
        r = yield call("code.create_repo", name=need.replace(".", "-"), owner=org, description=NEEDS[need].title)
        repo = r["result"]["repo"]
        yield call("code.commit", repo=repo, files=service_files(need, tier_source(need, best_tier(need)), price),
                   message="first version")
        inp, _ = NEEDS[need].generate(self.rng, 0.5)
        yield call("code.run", repo=repo, target="service:solve", input=inp)
        svc = yield from self.launch_service(org, acct, repo, f"{self.handle[:12]}-{need.split('.')[1][:6]}", need, price)
        if svc is None:
            return
        if self.advertise:
            yield call("ads.create_campaign", account=acct, keywords=list(NEEDS[need].keywords[:3]), bid_cents=1,
                       daily_budget_cents=3_000, title=f"{svc['host']}.crucible - better {need}",
                       text=f"{NEEDS[need].title}, verified quality", hosts=[svc["host"]], ref=svc["listing"] or "")
        yield call("forum.post", board="market", title=f"Launch: {svc['host']}.crucible",
                   body=f"We serve {need} at {price} cents per request. {svc['host']}.crucible")
        if self.license_price:
            yield call("market.list", kind="code", title=f"Licence: our {need} solver", price_cents=self.license_price,
                       seller=org, repo=repo, need=need,
                       description=f"Source licence for a verified {need} implementation ({NEEDS[need].title}).")
        yield from self.operate(org, acct, svc, repo, need)


class Employer(ScriptPolicy):
    name = "employer"

    def __init__(self, need: str = "route.shortest", capital: int = 1_500_000, salary: int = 600_000):
        self.need, self.capital, self.salary = need, capital, salary

    def script(self):
        need = self.need
        org, acct, _ = yield from self.found_company("works", self.capital)
        if acct is None:
            return
        r = yield call("code.create_repo", name=need.replace(".", "-"), owner=org)
        repo = r["result"]["repo"]
        r = yield call("issues.create_project", name="product", owner=org, repo=repo)
        project = r["result"]["project"]
        r = yield call("jobs.post", org=org, title="Software developer", role="developer",
                       salary_cents_month=self.salary, description=f"Build our {need} service.")
        posting = r["result"]["posting"]
        dev = None
        offered = set()
        yield from self.wait(24)          # let applications come in, then pick the strongest stated skill
        while dev is None:
            r = yield call("jobs.applications", posting=posting)
            apps = [a for a in (r["result"] if ok(r) else []) if a["status"] == "applied" and a["applicant"] not in offered]

            def stated_skill(a):
                import re
                m = re.search(r"skill ([0-9.]+)", a.get("note") or "")
                return float(m.group(1)) if m else 0.0
            if apps and not offered - {None}:
                a = max(apps, key=lambda a: (stated_skill(a), a["applicant"]))
                yield call("jobs.offer", application=a["application"], salary_cents_month=self.salary)
                offered.add(a["applicant"])
            r = yield call("id.profile", principal=org)
            members = [m["person"] for m in r["result"]["members"]] if ok(r) else []
            hired = [p for p in members if p in offered]
            if hired:
                dev = hired[0]
            else:
                yield from self.wait(6)
                r = yield call("contracts.list", status="rejected")
                rejected = {p for c in (r["result"] if ok(r) else []) for p in c["parties"] if p in offered}
                offered -= rejected          # declined offers free us to offer the next candidate
        r = yield call("issues.create", project=project, title=f"Implement the {need} solver",
                       body=f"Public format {need}. Commit a solve(args, caller) function.", labels=[f"need:{need}"],
                       assignee=dev)
        path = f"solutions/{need.replace('.', '_')}.py"
        while True:
            r = yield call("code.ls", repo=repo)
            if ok(r) and any(f["path"] == path for f in r["result"]["files"]):
                break
            yield from self.wait(12)
        r = yield call("market.browse", need=need, sort="price", limit=10)
        price = cheapest_rival(r["result"] if ok(r) else [], need, 0.95)
        yield call("code.commit", repo=repo, files=service_files(need, None, price, entry=path), message="manifest")
        svc = yield from self.launch_service(org, acct, repo, f"{self.handle[:12]}-{need.split('.')[1][:6]}", need, price)
        if svc is None:
            return
        yield from self.operate(org, acct, svc, repo, need)


SCORE_SRC = {
    "route.shortest": "def score(a, o):\n    w = {}\n    for u, v, c in a['edges']:\n        w[(u, v)] = min(c, w.get((u, v), 1e18)); w[(v, u)] = min(c, w.get((v, u), 1e18))\n"
                      "    p = o.get('path') or []\n    if not p or p[0] != a['source'] or p[-1] != a['target']: return -1e18\n"
                      "    s = 0\n    for x, y in zip(p, p[1:]):\n        if (x, y) not in w: return -1e18\n        s += w[(x, y)]\n    return -s\n",
    "schedule.jobs": "def score(a, o):\n    m = a['machines']; L = [0] * m\n    try:\n        for d, k in zip(a['durations'], o['assignment']): L[k] += d\n"
                     "    except Exception: return -1e18\n    return -max(L)\n",
    "pack.knapsack": "def score(a, o):\n    it = a['items']; ch = set(o.get('chosen') or [])\n    if sum(it[i][0] for i in ch) > a['capacity']: return -1e18\n"
                     "    return sum(it[i][1] for i in ch)\n",
    "sort.records": "def score(a, o):\n    return 1 if len(o.get('ids') or []) == len(a['records']) else -1e18\n",
}


class Composer(ScriptPolicy):
    """Builds on other firms' services: queries every known provider of a need and returns the best answer."""
    name = "composer"

    def __init__(self, need: str = "route.shortest", capital: int = 400_000, margin: float = 1.3):
        self.need, self.capital, self.margin = need, capital, margin

    def script(self):
        need = self.need
        r = yield call("market.browse", need=need, sort="price", limit=10)
        rivals = [x for x in (r["result"] if ok(r) else []) if x.get("host")]
        if not rivals:
            return
        hosts = [x["host"] for x in rivals[:3]]
        org, acct, _ = yield from self.found_company("mesh", self.capital)
        if acct is None:
            return
        upstream = {}
        for h in hosts:
            r = yield call("world.describe", host=h)
            if ok(r):
                for e in r["result"]["endpoints"]:
                    if e["name"].endswith(".solve"):
                        import re
                        m = re.search(r"\[price: (\d+) cents", e.get("description", ""))
                        upstream[h] = int(m.group(1)) if m else 0
        cost = sum(upstream.values())
        price = max(cost + 1, int(cost * self.margin) + 1)
        src = (f"import crucible_sdk\nHOSTS = {sorted(upstream)!r}\n" + SCORE_SRC.get(need, SCORE_SRC["sort.records"]) +
               "def solve(args, caller):\n    best, bs = None, -1e19\n    for h in HOSTS:\n        try:\n"
               "            r = crucible_sdk.call(h + '.solve', **args)\n        except crucible_sdk.WorldError:\n"
               "            continue\n        if r.get('status') != 200:\n            continue\n"
               "        s = score(args, r['body'])\n        if s > bs:\n            best, bs = r['body'], s\n"
               "    if best is None:\n        raise RuntimeError('no upstream answered')\n    return best\n")
        r = yield call("code.create_repo", name=f"{need.replace('.', '-')}-mesh", owner=org)
        repo = r["result"]["repo"]
        yield call("code.commit", repo=repo, files=service_files(need, src, price), message="aggregator")
        host = f"{self.handle[:12]}-mesh"
        r = yield call("cloud.providers")
        provs = sorted((p for p in (r or {}).get("result", []) if p["cores_free"] >= 1),
                       key=lambda p: (p["price_core_hour_cents"], p["provider"]))
        r = yield call("cloud.rent", provider=provs[0]["provider"], cores=1, account=acct, owner=org)
        inst = r["result"]["instance"]
        r = yield call("cloud.deploy", repo=repo, host=host, instance=inst)
        # the service principal pays upstream providers from the company account, within a monthly limit
        yield call("id.grant", grantee=f"svc:{host}", cap="bank.spend", owner=org, limit=int(self.capital // 2))
        r = yield call("market.list", kind="api", title=f"{host}: best-of {need}", price_cents=price, seller=org,
                       host=host, endpoint="solve", need=need,
                       description=f"Best answer from several providers. Keywords: {' '.join(NEEDS[need].keywords)}")
        svc = {"instance": inst, "listing": r["result"]["listing"] if ok(r) else None, "host": host}
        yield from self.operate(org, acct, svc, repo, need)


class BountyHunter(ScriptPolicy):
    name = "bounty_hunter"
    kind = "heuristic"

    def script(self):
        repo = None
        done = set()
        while True:
            r = yield call("jobs.bounties")
            for b in (r["result"] if ok(r) else []):
                if b["bounty"] in done or b["need"] not in NEEDS:
                    continue
                if repo is None:
                    r2 = yield call("code.create_repo", name="solvers")
                    if not ok(r2):
                        break
                    repo = r2["result"]["repo"]
                path = f"{b['need'].replace('.', '_')}.py"
                yield call("code.commit", repo=repo, files={path: tier_source(b["need"], best_tier(b["need"]))})
                yield call("jobs.submit", bounty=b["bounty"], repo=repo, target=f"{path[:-3]}:solve")
                done.add(b["bounty"])
                break
            yield from self.wait(24)
