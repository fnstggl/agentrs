"""Simulator-developed web applications: schema-first generation, installation and seeding (module `webgen`).

A *site spec* is data:

    {"name": "tooltrade", "title": "Tool Trade", "about": "...", "login": "optional"|"required",
     "entities": [{"name": "listing", "label": "Listing", "plural": "listings",
                   "fields": [{"name": "title", "type": "text", "required": true}, {"name": "price", "type": "money"},
                              {"name": "kind", "type": "choice", "options": ["drill", "saw"]}],
                   "statuses": ["open", "reserved", "closed"], "sellable": true}]}

`generate_spec(rng)` draws specs procedurally (themes x structure); `app_files(spec)` produces a complete,
self-contained application (manifest + one Python module with a generic, data-driven implementation of the spec):
HTML pages (home, list with search and pagination, detail, create form, about), form validation with error pages,
owner-only status transitions, purchases billed through bank invoices with payment status checked at the bank, a
JSON API, links between pages and to related sites, `noindex` on private pages, and per-site database storage.
Optional: `codegen` may be any callable spec -> files (e.g. a coding LLM); its output is accepted only if
`crucible.web.critic` passes it.

`webgen` installs accepted specs at genesis as sites operated by simulated organisations and seeds content by
calling the sites' own routes as their owners - so installation itself exercises them.
"""
from __future__ import annotations

import json

import numpy as np

from ..core.kernel import Module, register_module

THEMES = [
    ("tool library", "toolshed", [("tool", ["drill", "saw", "ladder", "sander"])], "borrow and lend tools"),
    ("community events", "eventboard", [("event", ["talk", "workshop", "meetup", "market"])], "local events"),
    ("used bikes", "bikebay", [("bike", ["road", "city", "cargo", "kids"])], "buy and sell bikes"),
    ("recipe exchange", "recipebox", [("recipe", ["soup", "bread", "salad", "dessert"])], "share recipes"),
    ("study groups", "studycircle", [("group", ["math", "languages", "coding", "history"])], "find study partners"),
    ("garden plots", "plotshare", [("plot", ["sunny", "shaded", "raised", "greenhouse"])], "rent garden plots"),
    ("repair cafe", "fixit", [("request", ["phone", "laptop", "lamp", "toaster"])], "get things repaired"),
    ("freelance gigs", "gigboard", [("gig", ["writing", "design", "data", "support"])], "post and find gigs"),
    ("lost and found", "foundit", [("item", ["keys", "wallet", "phone", "pet"])], "report lost and found items"),
    ("room rentals", "roomhop", [("room", ["single", "double", "studio", "shared"])], "rent rooms"),
    ("q and a", "askaway", [("question", ["tech", "legal", "health", "money"])], "ask and answer"),
    ("local produce", "farmstand", [("product", ["fruit", "vegetables", "eggs", "honey"])], "buy local produce"),
]
EXTRA_FIELDS = [("description", "longtext"), ("location", "text"), ("quantity", "int"), ("contact", "text"),
                ("day", "int"), ("condition", "choice:new,good,worn")]


def generate_spec(rng: np.random.Generator, idx: int = 0) -> dict:
    theme, stem, ents, about = THEMES[int(rng.integers(0, len(THEMES)))]
    name = f"{stem}{idx}" if idx else stem
    entities = []
    for ename, kinds in ents:
        fields = [{"name": "title", "type": "text", "required": True},
                  {"name": "kind", "type": "choice", "options": list(kinds), "required": True}]
        for fname, ftype in [EXTRA_FIELDS[i] for i in sorted(rng.choice(len(EXTRA_FIELDS), size=int(rng.integers(1, 4)),
                                                                         replace=False))]:
            f = {"name": fname, "type": ftype.split(":")[0], "required": bool(rng.random() < 0.3)}
            if ftype.startswith("choice:"):
                f["options"] = ftype.split(":")[1].split(",")
            fields.append(f)
        sellable = bool(rng.random() < 0.5)
        if sellable:
            fields.append({"name": "price", "type": "money", "required": True})
        entities.append({"name": ename, "label": ename.capitalize(), "plural": ename + "s", "fields": fields,
                         "statuses": ["open", "closed"] if rng.random() < 0.5 else ["open", "in progress", "done"],
                         "sellable": sellable})
    return {"name": name, "title": theme.title(), "about": f"A site to {about}.",
            "login": "required" if rng.random() < 0.2 else "optional", "entities": entities, "page_size": 10}


APP = r'''
import html
import json
import crucible_sdk as sdk

SPEC = json.loads(SPEC_JSON)
ENT = {e["name"]: e for e in SPEC["entities"]}
E = html.escape


def _page(title, body, req, noindex=False):
    user = req.get("user")
    who = ('Logged in as %s' % E(user)) if user else '<a href="/_login?return=%s">Log in</a>' % E(req["path"])
    nav = " | ".join('<a href="/%s">%s</a>' % (e["plural"], E(e["label"] + "s")) for e in SPEC["entities"])
    meta = '<meta name="robots" content="noindex">' if noindex else ""
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>" + meta + "<title>" + E(title) +
            "</title></head><body><header><a href='/'>" + E(SPEC["title"]) + "</a> | " + nav +
            " | <a href='/about'>About</a> | " + who + "</header><main>" + body + "</main></body></html>")


def _err(req, status, msg):
    return {"status": status, "html": _page("Error", "<p role='alert'>" + E(msg) + "</p>", req)}


def home(req, caller):
    parts = []
    for e in SPEC["entities"]:
        rows = sdk.call("db.query", collection=e["name"], order_by="t", desc=True, limit=5)
        lis = "".join('<li><a href="/%s/%s">%s</a></li>' % (e["plural"], r["id"], E(str(r.get("title", "")))) for r in rows)
        parts.append("<section><h2>Latest %ss</h2><ul>%s</ul><a href='/%s'>All %ss</a></section>" %
                     (E(e["label"]), lis or "<li>none yet</li>", e["plural"], E(e["label"])))
    return {"html": _page(SPEC["title"], "<h1>" + E(SPEC["title"]) + "</h1><p>" + E(SPEC["about"]) + "</p>" +
                          "".join(parts), req)}


def about(req, caller):
    return {"html": _page("About " + SPEC["title"], "<h1>About</h1><p>" + E(SPEC["about"]) + "</p><p>API: " +
                          ", ".join("/api/" + e["plural"] for e in SPEC["entities"]) + "</p>", req)}


def _ent(req):
    plural = req["path"].split("/")[1] if not req["path"].startswith("/api/") else req["path"].split("/")[2]
    for e in SPEC["entities"]:
        if e["plural"] == plural:
            return e
    return None


def listing(req, caller):
    e = _ent(req)
    q = (req["query"].get("q") or "").lower()
    status = req["query"].get("status") or ""
    page = max(1, int(req["query"].get("page") or 1)) if str(req["query"].get("page") or "1").isdigit() else 1
    rows = sdk.call("db.query", collection=e["name"], order_by="t", desc=True, limit=500)
    if q:
        rows = [r for r in rows if q in json.dumps(r).lower()]
    if status:
        rows = [r for r in rows if r.get("status") == status]
    n = SPEC.get("page_size", 10)
    shown = rows[(page - 1) * n: page * n]
    lis = "".join('<li><a href="/%s/%s">%s</a> (%s)</li>' % (e["plural"], r["id"], E(str(r.get("title", ""))),
                                                            E(r.get("status", ""))) for r in shown)
    pager = ""
    if page > 1:
        pager += '<a href="/%s?page=%d&q=%s">previous</a> ' % (e["plural"], page - 1, E(q))
    if page * n < len(rows):
        pager += '<a href="/%s?page=%d&q=%s">next</a>' % (e["plural"], page + 1, E(q))
    form = ('<form method="get" action="/%s"><label for="q">Search</label><input id="q" name="q" value="%s">'
            '<button type="submit">Search</button></form>' % (e["plural"], E(q)))
    new = '<p><a href="/%s/new">New %s</a></p>' % (e["plural"], E(e["label"].lower()))
    return {"html": _page(e["label"] + "s", "<h1>%ss</h1>%s%s<ul>%s</ul>%s<p>%d found</p>" %
                          (E(e["label"]), form, new, lis or "<li>nothing found</li>", pager, len(rows)), req)}


def _input(f, value=""):
    fid = "f-" + f["name"]
    req = " required" if f.get("required") else ""
    label = '<label for="%s">%s%s</label>' % (fid, E(f["name"].capitalize()), " *" if f.get("required") else "")
    if f["type"] == "choice":
        opts = "".join('<option value="%s">%s</option>' % (E(o), E(o)) for o in f["options"])
        return label + '<select id="%s" name="%s"%s>%s</select>' % (fid, f["name"], req, opts)
    if f["type"] == "longtext":
        return label + '<textarea id="%s" name="%s"%s>%s</textarea>' % (fid, f["name"], req, E(value))
    kind = "number" if f["type"] in ("int", "money") else "text"
    return label + '<input id="%s" name="%s" type="%s"%s value="%s">' % (fid, f["name"], kind, req, E(value))


def new_form(req, caller):
    e = _ent(req)
    if SPEC["login"] == "required" and not req["user"]:
        return {"status": 401, "html": _page("Log in", "<p>Log in to post.</p>", req, noindex=True)}
    inputs = "".join("<p>" + _input(f) + "</p>" for f in e["fields"])
    return {"html": _page("New " + e["label"].lower(), '<h1>New %s</h1><form method="post" action="/%s">%s'
                          '<button type="submit">Create</button></form>' % (E(e["label"].lower()), e["plural"], inputs),
                          req, noindex=True)}


def _validate(e, form):
    doc = {}
    for f in e["fields"]:
        raw = (form.get(f["name"]) or "").strip() if isinstance(form.get(f["name"]), str) else form.get(f["name"])
        if raw in (None, ""):
            if f.get("required"):
                return None, "%s is required" % f["name"]
            continue
        if f["type"] in ("int", "money"):
            try:
                v = int(float(raw))
            except (TypeError, ValueError):
                return None, "%s must be a number" % f["name"]
            if v < 0 or v > 10 ** 9:
                return None, "%s is out of range" % f["name"]
            doc[f["name"]] = v
        elif f["type"] == "choice":
            if raw not in f["options"]:
                return None, "%s must be one of %s" % (f["name"], ", ".join(f["options"]))
            doc[f["name"]] = raw
        else:
            doc[f["name"]] = str(raw)[:2000 if f["type"] == "longtext" else 200]
    return doc, None


def create(req, caller):
    e = _ent(req)
    if not req["user"]:
        return _err(req, 401, "Log in to create a %s." % e["label"].lower())
    doc, err = _validate(e, req["form"] if req["form"] else (req["json"] or {}))
    if err:
        return _err(req, 400, err)
    doc.update(owner=req["user"], status=e["statuses"][0], t=req["now"])
    r = sdk.call("db.put", collection=e["name"], doc=doc)
    if req["json"] is not None and not req["form"]:
        return {"status": 201, "json": {"id": r["id"], **doc}}
    return {"redirect": "/%s/%s" % (e["plural"], r["id"])}


def detail(req, caller):
    e = _ent(req)
    r = sdk.call("db.get", collection=e["name"], id=req["params"]["id"])
    if r is None:
        return _err(req, 404, "No such %s." % e["label"].lower())
    rows = "".join("<tr><th>%s</th><td>%s</td></tr>" % (E(f["name"]), E(str(r.get(f["name"], "")))) for f in e["fields"])
    body = "<h1>%s</h1><table>%s<tr><th>status</th><td id='status'>%s</td></tr><tr><th>owner</th><td>%s</td></tr>" \
           "</table>" % (E(str(r.get("title", ""))), rows, E(r.get("status", "")), E(r.get("owner", "")))
    if req["user"] == r.get("owner"):
        for s in e["statuses"]:
            if s != r.get("status"):
                body += ('<form method="post" action="/%s/%s/status"><input type="hidden" name="status" value="%s">'
                         '<button type="submit">Mark %s</button></form>' % (e["plural"], r["id"], E(s), E(s)))
    elif e.get("sellable") and r.get("status") == e["statuses"][0]:
        body += ('<form method="post" action="/%s/%s/buy"><button type="submit">Buy for $%.2f</button></form>' %
                 (e["plural"], r["id"], r.get("price", 0) / 100))
    body += '<p><a href="/%s">back</a></p>' % e["plural"]
    return {"html": _page(str(r.get("title", "")), body, req)}


def set_status(req, caller):
    e = _ent(req)
    r = sdk.call("db.get", collection=e["name"], id=req["params"]["id"])
    if r is None:
        return _err(req, 404, "No such item.")
    if req["user"] != r.get("owner"):
        return _err(req, 403, "Only the owner can change the status.")
    s = req["form"].get("status")
    if s not in e["statuses"]:
        return _err(req, 400, "Unknown status.")
    r["status"] = s
    sdk.call("db.put", collection=e["name"], id=r.pop("id"), doc=r)
    return {"redirect": "/%s/%s" % (e["plural"], req["params"]["id"])}


def buy(req, caller):
    e = _ent(req)
    if not req["user"]:
        return _err(req, 401, "Log in to buy.")
    r = sdk.call("db.get", collection=e["name"], id=req["params"]["id"])
    if r is None or not e.get("sellable"):
        return _err(req, 404, "Not for sale.")
    if r.get("status") != e["statuses"][0]:
        return _err(req, 409, "No longer available.")
    if req["user"] == r.get("owner"):
        return _err(req, 400, "You cannot buy your own item.")
    inv = sdk.call("bank.invoice", payer=req["user"], to_account=SPEC_ACCOUNT, amount=max(1, int(r.get("price", 1))),
                   memo="%s %s" % (SPEC["name"], r["id"]), ref="%s:%s" % (e["name"], r["id"]))
    sdk.call("db.put", collection="orders", doc={"item": r["id"], "entity": e["name"], "buyer": req["user"],
                                                 "invoice": inv["invoice"], "t": req["now"]})
    return {"html": _page("Order placed", "<p id='order'>Order placed. Pay invoice <b>%s</b> at bank.crucible; "
                          "your order is confirmed when it is paid.</p><p><a href='/orders'>Your orders</a></p>" %
                          E(inv["invoice"]), req, noindex=True)}


def orders(req, caller):
    if not req["user"]:
        return _err(req, 401, "Log in to see your orders.")
    rows = sdk.call("db.query", collection="orders", where={"buyer": req["user"]}, limit=100)
    lis = ""
    for o in rows:
        st = sdk.call("bank.invoice_status", invoice=o["invoice"])["status"]
        lis += "<li>%s %s: invoice %s (%s)</li>" % (E(o["entity"]), E(o["item"]), E(o["invoice"]), E(st))
    return {"html": _page("Your orders", "<h1>Your orders</h1><ul>%s</ul>" % (lis or "<li>none</li>"), req, noindex=True)}


def api_list(req, caller):
    e = _ent(req)
    return {"json": sdk.call("db.query", collection=e["name"], order_by="t", desc=True, limit=200)}


def api_create(req, caller):
    return create(req, caller)
'''


def app_files(spec: dict, operator_account: str) -> dict[str, str]:
    routes = {"GET /": "home", "GET /about": "about", "GET /orders": "orders"}
    for e in spec["entities"]:
        p = e["plural"]
        routes.update({f"GET /{p}": "listing", f"GET /{p}/new": "new_form", f"POST /{p}": "create",
                       f"GET /{p}/{{id}}": "detail", f"POST /{p}/{{id}}/status": "set_status",
                       f"GET /api/{p}": "api_list", f"POST /api/{p}": "api_create"})
        if e.get("sellable"):
            routes[f"POST /{p}/{{id}}/buy"] = "buy"
    manifest = {"entry": "service.py", "description": spec["about"],
                "web": {"title": spec["title"], "routes": routes, "rate_per_min": 240}}
    src = APP.replace("SPEC_JSON", repr(json.dumps(spec))).replace("SPEC_ACCOUNT", repr(operator_account))
    return {"crucible.json": json.dumps(manifest, indent=1), "service.py": src,
            "README.md": f"# {spec['title']}\n\n{spec['about']}\n\nGenerated from a site spec.\n"}


def sample_value(f: dict, rng) -> str:
    if f["type"] == "choice":
        return str(f["options"][int(rng.integers(0, len(f["options"])))])
    if f["type"] in ("int", "money"):
        return str(int(rng.integers(1, 5000)))
    words = ["solid", "blue", "small", "quiet", "vintage", "fresh", "handy", "bright", "local", "spare"]
    return " ".join(str(words[int(rng.integers(0, len(words)))]) for _ in range(int(rng.integers(2, 5))))


@register_module
class WebGen(Module):
    """Install generated sites at genesis, each operated by a simulated organisation, and seed them with content
    by posting through the sites' own routes. Config: {"sites": n} or {"specs": [...]}."""
    name = "webgen"
    depends = ("identity", "bank", "business", "code", "cloud", "net", "appdb")

    def genesis(self, ctx):
        cfg = self.config
        w = ctx.world
        rng = w.rng_for("webgen")
        specs = list(cfg.get("specs") or [generate_spec(rng, i) for i in range(int(cfg.get("sites", 0)))])
        if not specs:
            return
        idm, bank, biz, cloud = (w.modules[k] for k in ("identity", "bank", "business", "cloud"))
        providers = sorted(ctx.state.ids("providers"))
        installed = []
        for i, spec in enumerate(specs):
            owner = idm.create_person(ctx, f"{spec['name']}-admin", f"{spec['title']} admin", "npc", {"role": "webmaster"})
            bank.open_account(ctx, owner, "checking")
            oid = idm.create_org(ctx, spec["name"], spec["title"], owner, "company", {})
            biz.register_company(ctx, oid, owner, "web", fee=0)
            acct = bank.open_account(ctx, oid, "operating")
            bank.move(ctx, "acct_genesis", acct, int(cfg.get("cash", 400_000)), "endowment", f"capital of {oid}")
            code = w.modules["code"]
            rid = code.create_repo(ctx, oid, spec["name"], "public", spec["about"], "open")
            code.commit_files(ctx, rid, app_files(spec, acct), "site", owner, trusted=True)
            inst = ctx.call("cloud.rent", {"provider": providers[i % len(providers)], "cores": 1, "account": acct,
                                           "owner": oid}, as_principal=owner)
            ctx.call("cloud.deploy", {"repo": rid, "host": spec["name"], "instance": inst["instance"]}, as_principal=owner)
            ctx.call("id.grant", {"grantee": f"svc:{spec['name']}", "cap": "bank.read", "owner": oid, "scope": "*"},
                     as_principal=owner)
            installed.append((spec, owner))
        net = w.modules["net"]
        for spec, owner in installed:      # seed: the operators post content through their own sites
            ctx.call("net.login", {"site": spec["name"]}, as_principal=owner)
            for e in spec["entities"]:
                for _ in range(int(cfg.get("seed_items", 4))):
                    form = {f["name"]: sample_value(f, rng) for f in e["fields"]}
                    ctx.run_as(owner, lambda sub, e=e, form=form, spec=spec: net.fetch(
                        sub, f"http://{spec['name']}.crucible/{e['plural']}", "POST", form, None, owner))
        ctx.emit("note", {"what": "webgen", "sites": [s["name"] for s, _ in installed]}, vis="internal")
