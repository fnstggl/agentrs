"""Web sites for population worlds (module `popsites`) and the site application template.

A site is an ordinary deployment on the synthetic internet: code in a repository (`code`), a rented instance
(`cloud`), served by `net.fetch` like any agent-built site. Its usefulness to a visitor is never a number: it is what
its code serves. `app_files(spec, account)` generates a complete application from a spec:

    {"name": "cookhub", "title": "Cooking Guide", "topic": "cooking", "behaviour": "rich",
     "articles": [{"title": ..., "body": ...}], "products": [{"name": ..., "price": 1299, "about": ...}],
     "links": ["bikebay", ...], "about": "..."}

Pages: `/` (articles, shop, related sites), `/a/{id}` (an article), `/shop`, `/p/{id}` (a product with "Buy for
$X"), `POST /p/{id}/buy` (needs a logged-in visitor; bills them through a `bank.invoice` and records the order in the
site's own database), `/about`. `behaviour` changes what the code does - it is how a site is good or bad:

* `rich`: full articles (150-400 words about its topic);
* `thin`: a handful of words per page;
* `broken`: every page handler raises (HTTP 500);
* `flaky`: handlers fail during some hours of the day (deterministic in the request time);
* `spam`: pages are keyword lists repeating the topic's words, with links to other sites;
* `offtopic`: the title promises one topic, the text is about another.

`popsites` installs sites at genesis, each operated by a simulated organisation with its own account; config
`{"sites": [spec, ...]}` (explicit) or `{"n": 24, "mix": {...}, "shop_share": 0.4, "links": 3, "articles": [6, 12]}`
(drawn with `draw_specs`). Entrepreneurs (class 3) deploy the same template through ordinary tools.
"""
from __future__ import annotations

import json

import numpy as np

from ..core.kernel import Module, register_module
from . import text

BEHAVIOURS = ("rich", "thin", "broken", "flaky", "spam", "offtopic")
CATEGORIES = ("guide", "news", "shop", "blog", "tools", "club")   # perceivable from a page's title (res.kind)
DEFAULT_MIX = {"rich": 0.45, "thin": 0.15, "broken": 0.1, "flaky": 0.1, "spam": 0.1, "offtopic": 0.1}
FILLER = ("the", "a", "and", "with", "for", "your", "this", "that", "when", "how", "most", "many", "every", "some",
          "people", "often", "really", "simple", "good", "best", "first", "next", "time", "way", "make", "find",
          "try", "start", "keep", "learn", "show", "help", "need", "work", "small", "big", "local", "new")
TEMPLATES = (
    "{F} {A} {B} {F2} {C}.", "Here is {F} {A} {F2} {B} {F3} {C} {D}.", "{F} {A} is {F2} {B}, and {C} {F3} {D}.",
    "Why {A} {F} {B}? {F2} {C} {F3} {D}.", "A {F} {A} {B} guide: {C}, {D} and {E}.",
)


def compose(rng: np.random.Generator, words: list[str], n_words: int) -> str:
    """Prose-like text of about `n_words` words drawn from `words` and filler words (deterministic in rng)."""
    out, count = [], 0
    while count < n_words:
        t = TEMPLATES[int(rng.integers(0, len(TEMPLATES)))]
        pick = {k: str(words[int(rng.integers(0, len(words)))]) for k in "ABCDE"}
        pick.update({k: str(FILLER[int(rng.integers(0, len(FILLER)))]) for k in ("F", "F2", "F3")})
        s = t.format(**pick)
        out.append(s[0].upper() + s[1:])
        count += len(s.split())
    return " ".join(out)


def make_spec(rng: np.random.Generator, name: str, topic: str, behaviour: str, *, category: str | None = None,
              n_articles: int = 8, n_products: int = 0, price_range=(300, 4000), links=(), offtopic: str | None = None
              ) -> dict:
    """A complete site spec (content written here, from the topic's words)."""
    if behaviour not in BEHAVIOURS:
        raise ValueError(f"unknown behaviour {behaviour!r}")
    words = list(text.TOPICS[topic])
    body_words = list(text.TOPICS[offtopic or topic]) if behaviour == "offtopic" else words
    category = category or ("shop" if n_products else CATEGORIES[int(rng.integers(0, len(CATEGORIES)))])
    title = f"{topic.capitalize()} {category.capitalize()}"
    arts = []
    if behaviour == "thin":
        n_articles = min(int(n_articles), 3)
    for k in range(int(n_articles)):
        n = int(rng.integers(150, 400)) if behaviour in ("rich", "flaky", "offtopic", "broken") else int(rng.integers(4, 10))
        ttl = " ".join(str(words[int(rng.integers(0, len(words)))]) for _ in range(1 if behaviour == "thin" else 3)
                       ).capitalize()
        if behaviour == "spam":
            body = " ".join(str(words[int(rng.integers(0, 3))]) for _ in range(120)) + " best cheap deal click now"
        else:
            body = compose(rng, body_words, n)
        arts.append({"title": ttl, "body": body})
    prods = []
    for k in range(int(n_products)):
        prods.append({"name": f"{words[int(rng.integers(0, len(words)))]} {('kit', 'set', 'pack', 'box')[k % 4]}".title(),
                      "price": int(rng.integers(price_range[0], price_range[1] + 1)),
                      "about": compose(rng, words, int(rng.integers(20, 50))) if behaviour != "thin" else words[0]})
    about = {"thin": topic, "spam": " ".join(str(words[int(rng.integers(0, 3))]) for _ in range(60)) +
             " best cheap deal"}.get(behaviour) or compose(rng, words, 25)
    return {"name": name, "title": title, "topic": topic, "category": category, "behaviour": behaviour,
            "articles": arts, "products": prods, "links": list(links), "about": about}


APP = r'''
import html
import json

SPEC = json.loads(SPEC_JSON)
ACCOUNT = SPEC_ACCOUNT
E = html.escape
B = SPEC["behaviour"]


def _fail(req):
    if B == "broken":
        raise RuntimeError("internal server error")
    if B == "flaky" and (int(req["now"]) // 3600 + len(req["path"])) % 3 == 0:
        raise RuntimeError("upstream timeout")


def _page(title, body, req):
    nav = "<a href='/'>Home</a> | <a href='/about'>About</a>" + (" | <a href='/shop'>Shop</a>" if SPEC["products"] else "")
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><title>" + E(title) + "</title></head><body>"
            "<header>" + E(SPEC["title"]) + " | " + nav + "</header><main>" + body + "</main></body></html>")


def home(req, caller):
    _fail(req)
    arts = "".join("<li><a href='/a/%d'>%s</a></li>" % (i, E(a["title"])) for i, a in enumerate(SPEC["articles"]))
    rel = "".join("<li><a href='http://%s.crucible/'>%s</a></li>" % (E(h), E(h)) for h in SPEC["links"])
    shop = "<p><a href='/shop'>Visit the shop</a></p>" if SPEC["products"] else ""
    return {"html": _page(SPEC["title"], "<h1>" + E(SPEC["title"]) + "</h1><p>" + E(SPEC["about"]) + "</p><ul>" +
                          arts + "</ul>" + shop + ("<h2>Related sites</h2><ul>" + rel + "</ul>" if rel else ""), req)}


def article(req, caller):
    _fail(req)
    i = int(req["params"]["id"]) if str(req["params"]["id"]).isdigit() else -1
    if i < 0 or i >= len(SPEC["articles"]):
        return {"status": 404, "html": _page("Not found", "<p>No such article.</p>", req)}
    a = SPEC["articles"][i]
    nxt = "<p><a href='/a/%d'>Next article</a></p>" % ((i + 1) % len(SPEC["articles"]))
    return {"html": _page(a["title"], "<h1>" + E(a["title"]) + "</h1><p>" + E(a["body"]) + "</p>" + nxt, req)}


def shop(req, caller):
    _fail(req)
    lis = "".join("<li><a href='/p/%d'>%s</a> $%.2f</li>" % (i, E(p["name"]), p["price"] / 100)
                  for i, p in enumerate(SPEC["products"]))
    return {"html": _page("Shop", "<h1>Shop</h1><ul>" + (lis or "<li>nothing for sale</li>") + "</ul>", req)}


def product(req, caller):
    _fail(req)
    i = int(req["params"]["id"]) if str(req["params"]["id"]).isdigit() else -1
    if i < 0 or i >= len(SPEC["products"]):
        return {"status": 404, "html": _page("Not found", "<p>No such product.</p>", req)}
    p = SPEC["products"][i]
    form = "<form method='post' action='/p/%d/buy'><button type='submit'>Buy for $%.2f</button></form>" % (i, p["price"] / 100)
    return {"html": _page(p["name"], "<h1>" + E(p["name"]) + "</h1><p>" + E(p["about"]) + "</p>" + form, req)}


def buy(req, caller):
    _fail(req)
    import crucible_sdk as sdk
    i = int(req["params"]["id"]) if str(req["params"]["id"]).isdigit() else -1
    if i < 0 or i >= len(SPEC["products"]):
        return {"status": 404, "html": _page("Not found", "<p>No such product.</p>", req)}
    if not req.get("user"):
        return {"status": 401, "html": _page("Log in", "<p>Log in with your Crucible ID to buy.</p>", req)}
    p = SPEC["products"][i]
    inv = sdk.call("bank.invoice", payer=req["user"], to_account=ACCOUNT, amount=int(p["price"]),
                   memo="%s: %s" % (SPEC["name"], p["name"]), ref="%s:p%d" % (SPEC["name"], i))
    sdk.call("db.put", collection="orders", doc={"product": i, "buyer": req["user"], "invoice": inv["invoice"],
                                                 "price": int(p["price"]), "t": req["now"]})
    return {"html": _page("Order placed", "<p id='order'>Order placed. Pay invoice <b>%s</b> at the bank.</p>" %
                          E(inv["invoice"]), req)}


def about(req, caller):
    _fail(req)
    return {"html": _page("About " + SPEC["title"], "<h1>About</h1><p>" + E(SPEC["about"]) + "</p>", req)}
'''


def app_files(spec: dict, operator_account: str) -> dict[str, str]:
    routes = {"GET /": "home", "GET /about": "about", "GET /a/{id}": "article"}
    if spec.get("products"):
        routes.update({"GET /shop": "shop", "GET /p/{id}": "product", "POST /p/{id}/buy": "buy"})
    manifest = {"entry": "service.py", "description": spec.get("about", "")[:400],
                "web": {"title": spec["title"], "routes": routes, "rate_per_min": 600}}
    src = APP.replace("SPEC_JSON", repr(json.dumps(spec, sort_keys=True))).replace("SPEC_ACCOUNT", repr(operator_account))
    return {"crucible.json": json.dumps(manifest, indent=1), "service.py": src,
            "README.md": f"# {spec['title']}\n\n{spec.get('about', '')}\n"}


def draw_specs(rng: np.random.Generator, n: int, mix: dict | None = None, shop_share: float = 0.4, links: int = 3,
               articles=(6, 12), prefix: str = "") -> list[dict]:
    """`n` site specs with topics, categories and behaviours drawn from `mix`; rich sites link to related sites."""
    mix = dict(mix or DEFAULT_MIX)
    names_b = [b for b in BEHAVIOURS if mix.get(b, 0) > 0]
    p = np.array([mix[b] for b in names_b], float)
    p /= p.sum()
    topics = list(text.TOPIC_NAMES)
    specs = []
    for k in range(int(n)):
        topic = topics[int(rng.integers(0, len(topics)))]
        beh = names_b[int(rng.choice(len(names_b), p=p))]
        shop = bool(rng.random() < shop_share)
        specs.append({"k": k, "topic": topic, "behaviour": beh, "shop": shop})
    hosts = [f"{prefix}{s['topic'][:6]}{s['k']}" for s in specs]
    out = []
    for s, host in zip(specs, hosts):
        same = [h for h, t in zip(hosts, specs) if t["topic"] == s["topic"] and h != host]
        other = [h for h in hosts if h != host and h not in same]
        rng.shuffle(same)
        rng.shuffle(other)
        lk = (same + other)[: int(links)] if s["behaviour"] in ("rich", "spam", "flaky") else []
        off = topics[(topics.index(s["topic"]) + 5) % len(topics)]
        out.append(make_spec(rng, host, s["topic"], s["behaviour"], n_articles=int(rng.integers(articles[0], articles[1] + 1)),
                             n_products=int(rng.integers(2, 6)) if s["shop"] else 0, links=lk, offtopic=off))
    return out


@register_module
class PopSites(Module):
    """Install web sites at genesis, each operated by its own simulated organisation (an endowed account pays its
    hosting). Config: {"sites": [spec...]} or {"n", "mix", "shop_share", "links", "articles", "cash"}."""
    name = "popsites"
    depends = ("identity", "bank", "code", "cloud", "net", "appdb")

    def genesis(self, ctx):
        cfg = self.config
        w = ctx.world
        specs = list(cfg.get("sites") or draw_specs(w.rng_for("popsites"), int(cfg.get("n", 0)), cfg.get("mix"),
                                                    float(cfg.get("shop_share", 0.4)), int(cfg.get("links", 3)),
                                                    tuple(cfg.get("articles", (6, 12)))))
        if not specs:
            return
        install_sites(ctx, specs, cash=int(cfg.get("cash", 2_000_000)))


def ensure_provider(ctx) -> str:
    """A compute provider to rent from (an existing one, else one is founded)."""
    st = ctx.state
    provs = sorted(st.ids("providers"))
    if provs:
        return provs[0]
    w = ctx.world
    idm, bank, cloud = w.modules["identity"], w.modules["bank"], w.modules["cloud"]
    owner = idm.create_person(ctx, "popcloud-ceo", "Popcloud CEO", "npc", {"role": "owner"})
    bank.open_account(ctx, owner, "checking")
    oid = idm.create_org(ctx, "popcloud", "Popcloud Hosting", owner, "company", {})
    bank.open_account(ctx, oid, "operating")
    cloud.add_provider(ctx, oid, "Popcloud Hosting", 2, 4096, 0.999)
    return oid


def install_sites(ctx, specs: list[dict], cash: int = 2_000_000) -> list[str]:
    """Create operator, organisation, account, repository and deployment of each site spec; returns the hosts."""
    w = ctx.world
    idm, bank, code = w.modules["identity"], w.modules["bank"], w.modules["code"]
    provider = ensure_provider(ctx)
    hosts = []
    for spec in specs:
        handle = spec["name"]
        owner = idm.create_person(ctx, f"{handle}-op", f"{spec['title']} operator", "npc", {"role": "webmaster"})
        bank.open_account(ctx, owner, "checking")
        oid = idm.create_org(ctx, handle, spec["title"], owner, "company", {})
        acct = bank.open_account(ctx, oid, "operating")
        if cash:
            bank.move(ctx, "acct_genesis", acct, int(cash), "endowment", f"capital of {oid}")
        rid = code.create_repo(ctx, oid, handle, "public", spec.get("about", "")[:200], "open")
        code.commit_files(ctx, rid, app_files(spec, acct), "site", owner, trusted=True)
        inst = ctx.call("cloud.rent", {"provider": provider, "cores": 1, "account": acct, "owner": oid},
                        as_principal=owner)
        ctx.call("cloud.deploy", {"repo": rid, "host": handle, "instance": inst["instance"]}, as_principal=owner)
        ctx.call("id.grant", {"grantee": f"svc:{handle}", "cap": "bank.read", "owner": oid, "scope": "*"},
                 as_principal=owner)
        hosts.append(handle)
    ctx.emit("note", {"what": "popsites", "sites": [{"host": s["name"], "topic": s["topic"],
                                                      "behaviour": s["behaviour"], "category": s["category"],
                                                      "products": len(s["products"])} for s in specs]},
             vis="internal")
    return hosts
