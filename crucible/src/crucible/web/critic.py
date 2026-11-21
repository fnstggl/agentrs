"""Site critic: validate a web application on the synthetic internet by using it, the way a careful QA engineer
would - never by reading its code.

`api_critic(world, site, spec)` runs journeys through `net.fetch` as fresh test personas and checks the results
against authoritative world state: pages render, created records persist in the application database and appear
on list and detail pages, invalid input is rejected without side effects, logged-out visitors cannot write,
only owners can change status, purchases create real bank invoices whose payment the site then reports, the
JSON API agrees with the database, private pages carry `noindex`, and every link on the home page resolves.

`browser_critic(app_url, world, site, spec, handle, password)` repeats the main journey in Chromium: navigation,
labelled form controls (accessibility names), form submission, the visible result, the error state, and no
console errors.

A site is accepted only if every critical check passes; reports list every check with evidence.
"""
from __future__ import annotations

import json
import re

import numpy as np

HREF = re.compile(r'href="([^"]+)"')


class Report:
    def __init__(self, site):
        self.site = site
        self.checks: list[dict] = []

    def check(self, name, ok, critical=True, **evidence):
        self.checks.append({"check": name, "ok": bool(ok), "critical": critical, **evidence})
        return bool(ok)

    @property
    def accepted(self) -> bool:
        return all(c["ok"] for c in self.checks if c["critical"])

    def to_dict(self):
        return {"site": self.site, "accepted": self.accepted, "passed": sum(c["ok"] for c in self.checks),
                "total": len(self.checks), "checks": self.checks}


def _persona(world, handle, cash=500_000, password=None):
    if not world.state.has("principals", f"p:{handle}"):
        world.system_call("admin.join", {"handle": handle, "endowment": cash, **({"password": password} if password else {})})
    return f"p:{handle}"


def api_critic(world, site: str, spec: dict, seed: int = 0) -> dict:
    from ..worldgen.sitegen import sample_value
    rng = np.random.default_rng(seed)
    rep = Report(site)
    a = _persona(world, f"qa-{site[:12]}-a")
    b = _persona(world, f"qa-{site[:12]}-b")
    base = f"http://{site}.crucible"

    def fetch(who, path, method="GET", form=None, js=None):
        r = world.execute(who, {"tool": "net.fetch", "args": {"url": base + path, "method": method,
                                                              **({"form": form} if form is not None else {}),
                                                              **({"json": js} if js is not None else {})}})
        return r.result if r.ok else {"status": -1, "body": str(r.error), "content_type": ""}

    def ndocs(coll):
        return len(world.state.find_ids("app_docs", "nc", f"{site}|{coll}"))
    home = fetch(a, "/")
    rep.check("home renders", home["status"] == 200 and "<title>" in home["body"], status=home["status"])
    for href in sorted(set(HREF.findall(home["body"])))[:30]:
        if href.startswith("/") and not href.startswith("/_"):
            rep.check(f"link {href} resolves", fetch(a, href)["status"] in (200, 401), critical=False)
    world.execute(a, {"tool": "net.login", "args": {"site": site}})
    world.execute(b, {"tool": "net.login", "args": {"site": site}})
    for e in spec["entities"]:
        p = e["plural"]
        rep.check(f"{p}: list renders", fetch(a, f"/{p}")["status"] == 200)
        form = {f["name"]: sample_value(f, rng) for f in e["fields"]}
        n0 = ndocs(e["name"])
        r = fetch(a, f"/{p}", "POST", form)
        ok = rep.check(f"{p}: create redirects to the new record", r["status"] == 303 and r.get("location", "").startswith(f"/{p}/"),
                       status=r["status"])
        rep.check(f"{p}: create writes exactly one record", ndocs(e["name"]) == n0 + 1, before=n0, after=ndocs(e["name"]))
        if not ok:
            continue
        rid = r["location"].rsplit("/", 1)[-1]
        d = fetch(a, f"/{p}/{rid}")
        rep.check(f"{p}: detail shows the submitted values", d["status"] == 200 and all(
            str(v) in d["body"] or f["type"] in ("money",) for f in e["fields"] for k, v in form.items() if k == f["name"]))
        lst = fetch(b, f"/{p}?q={form['title'].split()[0]}")
        rep.check(f"{p}: record persists and is searchable by others", form["title"].split()[0] in lst["body"])
        req = [f for f in e["fields"] if f.get("required")]
        if req:
            bad = dict(form)
            bad.pop(req[0]["name"])
            n1 = ndocs(e["name"])
            r2 = fetch(a, f"/{p}", "POST", bad)
            rep.check(f"{p}: missing required field rejected without side effects",
                      r2["status"] == 400 and ndocs(e["name"]) == n1, status=r2["status"])
        anon = _persona(world, f"qa-{site[:12]}-anon")
        n2 = ndocs(e["name"])
        r3 = fetch(anon, f"/{p}", "POST", form)
        rep.check(f"{p}: logged-out visitor cannot create", r3["status"] in (401, 403) and ndocs(e["name"]) == n2,
                  status=r3["status"])
        other = [s for s in e["statuses"] if s != e["statuses"][0]][0]
        r4 = fetch(b, f"/{p}/{rid}/status", "POST", {"status": other})
        rep.check(f"{p}: non-owner cannot change status", r4["status"] == 403)
        if e.get("sellable"):
            inv0 = world.state.count("invoices")
            r6 = fetch(b, f"/{p}/{rid}/buy", "POST", {})
            created = world.state.count("invoices") == inv0 + 1
            rep.check(f"{p}: purchase creates a bank invoice for the buyer", r6["status"] == 200 and created)
            if created:
                inv = sorted(world.state.ids("invoices"))[-1]
                acct = world.modules["bank"].primary_account(world.state, b)
                paid = world.execute(b, {"tool": "bank.pay_invoice", "args": {"invoice": inv, "from_account": acct}}).ok
                o = fetch(b, "/orders")
                rep.check(f"{p}: paid order is reported paid by the site", paid and "paid" in o["body"], paid=paid)
        r5 = fetch(a, f"/{p}/{rid}/status", "POST", {"status": other})
        rep.check(f"{p}: owner can change status", r5["status"] == 303 and other in fetch(a, f"/{p}/{rid}")["body"])
        api = fetch(b, f"/api/{p}")
        try:
            rows = json.loads(api["body"])
            rep.check(f"{p}: JSON API agrees with the database", len(rows) == ndocs(e["name"]) or len(rows) == 200)
        except (TypeError, ValueError):
            rep.check(f"{p}: JSON API agrees with the database", False)
        nf = fetch(a, f"/{p}/new")
        rep.check(f"{p}: form page is noindex", "noindex" in nf["body"], critical=False)
    return rep.to_dict()


def browser_critic(app_url: str, world, site: str, spec: dict, handle: str, password: str, seed: int = 0) -> dict:
    from ..worldgen.sitegen import sample_value
    from .browser import BrowserSession
    rng = np.random.default_rng(seed + 1)
    rep = Report(site)
    b = BrowserSession(app_url, timeout_ms=20_000)
    try:
        b.act({"action": "goto", "url": "id.crucible"})
        b.act({"action": "fill", "target": "login-handle", "value": handle})
        b.act({"action": "fill", "target": "Password", "value": password})
        b.act({"action": "click", "target": "Sign in"})
        o = b.act({"action": "goto", "url": f"http://{site}.crucible/"})
        rep.check("browser: home renders with a title", o["ok"] and o["title"] == spec["title"], title=o.get("title"))
        e = spec["entities"][0]
        o = b.act({"action": "click", "target": "Log in"})                  # the site's own login link
        rep.check("browser: the site offers Crucible ID login", "wants to know who you are" in o["text"],
                  text=o["text"][:200])
        o = b.act({"action": "click", "target": f"Log in as p:{handle}"})
        rep.check("browser: after login the site greets the user", f"Logged in as p:{handle}" in o["text"])
        o = b.act({"action": "goto", "url": f"http://{site}.crucible/{e['plural']}/new"})
        ids = {el["id"]: el for el in o["elements"]}
        rep.check("browser: every form control has an accessible name",
                  all(ids.get(f"f-{f['name']}", {}).get("name") for f in e["fields"]),
                  missing=[f["name"] for f in e["fields"] if not ids.get(f"f-{f['name']}", {}).get("name")])
        vals = {f["name"]: sample_value(f, rng) for f in e["fields"]}
        for f in e["fields"]:
            if f["type"] == "choice":
                b.select(f"f-{f['name']}", vals[f["name"]])
            else:
                b.fill(f"f-{f['name']}", vals[f["name"]])
        o = b.act({"action": "click", "target": "Create"})
        rep.check("browser: submitting the form shows the new record", vals["title"] in o["text"] and
                  f"/{e['plural']}/" in o["url"], url=o["url"])
        o = b.act({"action": "click", "target": "back"})
        rep.check("browser: the record is listed after navigation", vals["title"] in o["text"])
        if any(f.get("required") and f["type"] not in ("choice",) for f in e["fields"]):
            b.act({"action": "goto", "url": f"http://{site}.crucible/{e['plural']}/new"})
            b.page.evaluate("() => document.querySelectorAll('[required]').forEach(x => x.removeAttribute('required'))")
            o = b.act({"action": "click", "target": "Create"})
            rep.check("browser: invalid input shows an error message", any(m["role"] == "alert" for m in o["messages"])
                      or "required" in o["text"])
        # HTTP error statuses are logged by Chromium as "Failed to load resource"; they are checked separately
        site_errors = [c for c in b.console_log if f"{site}.crucible" in (c["url"] or "")
                       and not c["text"].startswith("Failed to load resource")]
        rep.check("browser: no console errors from the site's scripts", not site_errors, errors=site_errors[:5])
        rep.check("browser: no other console errors", not b.console_errors, critical=False,
                  errors=b.console_log[:5])
        rep.check("browser: no request left the synthetic network", not b.blocked, blocked=b.blocked[:5])
    finally:
        b.close()
    return rep.to_dict()
