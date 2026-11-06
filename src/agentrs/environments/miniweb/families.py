"""MiniWeb task families.

Each family is a small web app with server-side state and a verifier. Families
share low-level dynamics (links navigate, typing sets values, submit validates,
menus expand, dialogs confirm) so that knowledge of action consequences learned
on some families can, in principle, transfer to held-out ones.
"""

from __future__ import annotations

import random
import re
from typing import Callable

from agentrs.environments.base import TaskSpec, Verification
from agentrs.environments.miniweb.core import App, Page, W

# ---------------------------------------------------------------------- vocab
# Disjoint vocabularies for train/val vs test instances (held-out surface forms).
_WORDS_ALL = [
    "Apple", "Birch", "Cobalt", "Delta", "Ember", "Falcon", "Garnet", "Harbor", "Indigo", "Juniper",
    "Kestrel", "Lumen", "Maple", "Nectar", "Onyx", "Pepper", "Quartz", "Raven", "Saffron", "Tundra",
    "Umber", "Velvet", "Willow", "Xenon", "Yarrow", "Zephyr", "Amber", "Basil", "Cedar", "Dune",
    "Echo", "Fern", "Glacier", "Hazel", "Iris", "Jasper", "Koala", "Lotus", "Mango", "Nova",
    "Olive", "Prism", "Quill", "Ridge", "Sable", "Topaz", "Ultra", "Vista", "Wren", "Yeti",
]
WORDS_TRAIN = _WORDS_ALL[0::2]
WORDS_TEST = _WORDS_ALL[1::2]

_PEOPLE_ALL = [
    "Alice Smith", "Bob Jones", "Carla Diaz", "Dev Patel", "Erin Moore", "Farid Khan", "Grace Lee",
    "Hugo Brown", "Ines Costa", "Jon Park", "Kara White", "Luis Gomez", "Mia Chen", "Nils Berg",
    "Omar Aziz", "Pia Rossi", "Quinn Hall", "Rosa Lima", "Sam Wood", "Tara Nash",
]
PEOPLE_TRAIN = _PEOPLE_ALL[0::2]
PEOPLE_TEST = _PEOPLE_ALL[1::2]

_CITIES_ALL = ["Paris", "Lagos", "Osaka", "Lima", "Oslo", "Cairo", "Quito", "Perth", "Dubai", "Hanoi",
               "Turin", "Accra", "Kyoto", "Porto", "Riga", "Seoul"]
CITIES_TRAIN = _CITIES_ALL[0::2]
CITIES_TEST = _CITIES_ALL[1::2]

_ADJ = ["Blue", "Red", "Green", "Silver", "Golden", "Tiny", "Large", "Smart", "Classic", "Modern"]
_NOUNS = ["Lamp", "Chair", "Kettle", "Backpack", "Clock", "Mug", "Speaker", "Desk", "Fan", "Notebook",
          "Pillow", "Bottle"]

SECTIONS = {
    "Billing": ["Invoices", "Payment methods", "Plans"],
    "Profile": ["Avatar", "Bio", "Contact info"],
    "Security": ["Password", "Two-factor", "Sessions"],
    "Help": ["FAQ", "Contact support", "Guides"],
    "Projects": ["Archived", "Shared", "Templates"],
    "Team": ["Members", "Roles", "Invites"],
}

SETTINGS = ["Dark mode", "Notifications", "Auto-save", "Location access", "Sounds", "Sync", "Beta features"]
FIELD_LABELS = ["Name", "Nickname", "City", "Code", "Comment", "Title", "Company", "Tag"]
MENU_ITEMS = ["Reply", "Forward", "Archive", "Delete", "Mark as spam", "Pin", "Snooze"]
DROPDOWNS = {
    "Shipping method": ["Standard", "Express", "Overnight", "Pickup"],
    "Size": ["Small", "Medium", "Large", "X-Large"],
    "Color": ["Black", "White", "Navy", "Olive", "Coral"],
    "Priority": ["Low", "Normal", "High", "Urgent"],
}


def vocab(split: str, train: list, test: list) -> list:
    return test if split == "test" else train


def product_names(rng: random.Random, n: int) -> list[str]:
    # ordered list (not a set): set iteration order depends on PYTHONHASHSEED and would make
    # task generation differ between processes
    names: list[str] = []
    while len(names) < n:
        x = f"{rng.choice(_ADJ)} {rng.choice(_NOUNS)}"
        if x not in names:
            names.append(x)
    return names


def price(rng: random.Random) -> str:
    return f"${rng.randint(3, 199)}.{rng.choice(['00', '50', '99', '25'])}"


def apply_layout(widgets: list[W], layout: str, rng: random.Random, keep_last: int = 0) -> list[W]:
    """Layout variants: 'default', 'reversed', 'distractors' (held out)."""
    if layout == "reversed":
        if keep_last:
            widgets = widgets[:-keep_last][::-1] + widgets[-keep_last:]
        else:
            widgets = widgets[::-1]
    elif layout == "distractors":
        extra = [
            W(key="dx_ad", role="text", name="Sponsored: try our new app!"),
            W(key="dx_link", role="link", name="Privacy policy", href="/privacy"),
            W(key="dx_note", role="text", name="Tip: changes are saved automatically."),
        ]
        widgets = extra[:2] + widgets + extra[2:]
    return widgets


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


# ======================================================================= families
class ClickButton(App):
    family, category = "click_button", "single_action"
    TEMPLATES = ['Click the "{t}" button.', 'Press the button labeled "{t}".', 'Find and click "{t}".']

    @classmethod
    def make(cls, rng: random.Random, split: str) -> dict:
        labels = rng.sample(vocab(split, WORDS_TRAIN, WORDS_TEST), rng.randint(3, 6))
        t = rng.choice(labels)
        return {"labels": labels, "target": t, "t": t}

    def setup(self):
        self.url = "/"

    def _click(self, label):
        self.server["clicked"] = label
        self.terminal = True

    def render(self) -> Page:
        p = self.spec.private
        ws = [self.text("intro", "Choose an option below.")]
        for i, lab in enumerate(p["labels"]):
            ws.append(self.button(f"b{i}", lab, lambda lab=lab: self._click(lab)))
        return Page("/", "Buttons", apply_layout(ws, self.spec.layout, self.rng))

    def verify(self) -> Verification:
        ok = self.server.get("clicked") == self.spec.private["target"]
        return Verification(success=ok, score=float(ok), progress=float(ok),
                            evidence={"clicked": self.server.get("clicked")})


class NavigateLink(App):
    family, category = "navigate_link", "navigation"
    allowed_paths = ("/", "/s/*")
    TEMPLATES = ['Open the "{sub}" page under "{sec}", then finish.',
                 'Go to {sec} > "{sub}" and finish when you are there.',
                 'Navigate to the "{sub}" page in the "{sec}" section, then finish.']

    @classmethod
    def make(cls, rng, split):
        secs = rng.sample(sorted(SECTIONS), 4)
        sec = rng.choice(secs)
        sub = rng.choice(SECTIONS[sec])
        return {"sections": secs, "sec": sec, "sub": sub,
                "target_url": f"/s/{sec.lower()}/{sub.lower().replace(' ', '-')}"}

    def render(self) -> Page:
        p = self.spec.private
        u = self.url
        if u == "/":
            ws = [self.text("h", "Dashboard", role="heading")]
            ws += [self.link(f"l{i}", s, f"/s/{s.lower()}") for i, s in enumerate(p["sections"])]
            return Page("/", "Dashboard", apply_layout(ws, self.spec.layout, self.rng))
        parts = u.strip("/").split("/")
        if len(parts) == 2:
            sec = next((s for s in SECTIONS if s.lower() == parts[1]), None)
            if sec is None:
                return Page(u, "Not found", [self.text("nf", "Page not found"), self.link("home", "Home", "/")])
            ws = [self.text("h", sec, role="heading"), self.link("home", "Home", "/")]
            ws += [self.link(f"l{i}", s, f"/s/{parts[1]}/{s.lower().replace(' ', '-')}")
                   for i, s in enumerate(SECTIONS[sec])]
            return Page(u, sec, apply_layout(ws, self.spec.layout, self.rng))
        if len(parts) == 3:
            sec = next((s for s in SECTIONS if s.lower() == parts[1]), "?")
            sub = next((s for s in SECTIONS.get(sec, []) if s.lower().replace(" ", "-") == parts[2]), "?")
            ws = [self.text("h", sub, role="heading"), self.text("t", f"This is the {sub} page of {sec}."),
                  self.link("up", sec, f"/s/{parts[1]}"), self.link("home", "Home", "/")]
            return Page(u, f"{sec} - {sub}", ws)
        return Page(u, "Not found", [self.text("nf", "Page not found"), self.link("home", "Home", "/")])

    def verify(self):
        p = self.spec.private
        at = self.url == p["target_url"]
        ok = at and self.server.get("finished", False)
        prog = 1.0 if at else (0.5 if self.url.startswith(f"/s/{p['sec'].lower()}") else 0.0)
        return Verification(success=ok, score=float(ok), progress=prog, evidence={"url": self.url})


class FillField(App):
    family, category = "fill_field", "form"
    TEMPLATES = ['Enter "{v}" into the {label} field and submit.',
                 'Type "{v}" in the {label} box, then press Submit.',
                 'Fill {label} with "{v}" and submit the form.']

    @classmethod
    def make(cls, rng, split):
        labels = rng.sample(FIELD_LABELS, rng.randint(1, 3))
        label = rng.choice(labels)
        v = rng.choice(vocab(split, WORDS_TRAIN, WORDS_TEST)).lower() + str(rng.randint(1, 99))
        return {"labels": labels, "label": label, "v": v}

    def submit(self):
        p = self.spec.private
        self.server["submitted"] = {lab: self.field_values.get(f"f{i}", "") for i, lab in enumerate(p["labels"])}
        self.terminal = True

    def render(self):
        p = self.spec.private
        ws = [self.textbox(f"f{i}", lab) for i, lab in enumerate(p["labels"])]
        ws = apply_layout(ws, self.spec.layout, self.rng)
        ws.append(self.button("submit", "Submit", self.submit))
        return Page("/form", "Form", ws)

    def verify(self):
        p = self.spec.private
        sub = self.server.get("submitted") or {}
        ok = _norm(sub.get(p["label"])) == _norm(p["v"])
        return Verification(success=ok, score=float(ok), progress=float(ok), evidence={"submitted": sub})


class Login(App):
    family, category = "login", "form"
    TEMPLATES = ['Log in with username "{u}" and password "{pw}".',
                 'Sign in as "{u}" using the password "{pw}".',
                 'Use the credentials "{u}" / "{pw}" to log in.']

    @classmethod
    def make(cls, rng, split):
        u = rng.choice(vocab(split, PEOPLE_TRAIN, PEOPLE_TEST)).split()[0].lower() + str(rng.randint(1, 9))
        pw = rng.choice(vocab(split, WORDS_TRAIN, WORDS_TEST)).lower() + "#" + str(rng.randint(10, 99))
        return {"u": u, "pw": pw}

    def setup(self):
        self.url = "/login"
        self.server["failed_attempts"] = 0

    def sign_in(self):
        p = self.spec.private
        if self.field_values.get("user") == p["u"] and self.field_values.get("pw") == p["pw"]:
            self.server["logged_in"] = True
            self.goto("/home")
            self.terminal = True
        else:
            self.server["failed_attempts"] += 1
            self.server["error"] = "Invalid username or password"

    def render(self):
        if self.url == "/home":
            return Page("/home", "Home", [self.text("w", f"Welcome back, {self.spec.private['u']}!",
                                                    role="heading")])
        ws = [self.text("h", "Sign in", role="heading"), self.textbox("user", "Username"),
              self.textbox("pw", "Password")]
        ws = apply_layout(ws, self.spec.layout, self.rng)
        if self.server.get("error"):
            ws.append(self.text("err", self.server["error"], role="alert"))
        ws.append(self.button("signin", "Sign in", self.sign_in))
        return Page("/login", "Sign in", ws)

    def verify(self):
        ok = bool(self.server.get("logged_in"))
        return Verification(success=ok, score=float(ok), progress=float(ok),
                            evidence={"failed_attempts": self.server.get("failed_attempts")})


class Checkboxes(App):
    family, category = "checkboxes", "form"
    TEMPLATES = ['Select {items}, then submit.', 'Check {items} and press Submit.',
                 'Tick exactly {items} and submit.']

    @classmethod
    def make(cls, rng, split):
        labels = rng.sample(vocab(split, WORDS_TRAIN, WORDS_TEST), rng.randint(3, 6))
        k = rng.randint(1, min(3, len(labels)))
        targets = rng.sample(labels, k)
        return {"labels": labels, "targets": sorted(targets),
                "items": " and ".join(f'"{t}"' for t in targets)}

    def submit(self):
        p = self.spec.private
        self.server["submitted"] = sorted(lab for i, lab in enumerate(p["labels"]) if self.checks.get(f"c{i}"))
        self.terminal = True

    def render(self):
        p = self.spec.private
        ws = [self.checkbox(f"c{i}", lab) for i, lab in enumerate(p["labels"])]
        ws = apply_layout(ws, self.spec.layout, self.rng)
        ws.append(self.button("submit", "Submit", self.submit))
        return Page("/", "Checkboxes", ws)

    def verify(self):
        p = self.spec.private
        sub = self.server.get("submitted")
        ok = sub == p["targets"]
        cur = {lab for i, lab in enumerate(p["labels"]) if self.checks.get(f"c{i}")}
        tgt = set(p["targets"])
        prog = len(cur & tgt) / max(1, len(tgt | cur))
        return Verification(success=ok, score=float(ok), progress=1.0 if ok else prog, evidence={"submitted": sub})


class Dropdown(App):
    family, category = "dropdown", "form"
    TEMPLATES = ['Choose "{opt}" from the {label} menu and submit.',
                 'Set {label} to "{opt}" and press Submit.',
                 'Pick "{opt}" as the {label}, then submit.']

    @classmethod
    def make(cls, rng, split):
        label = rng.choice(sorted(DROPDOWNS))
        opts = DROPDOWNS[label][:]
        rng.shuffle(opts)
        return {"label": label, "options": opts, "opt": rng.choice(opts)}

    def submit(self):
        self.server["submitted"] = self.field_values.get("dd", "")
        self.terminal = True

    def render(self):
        p = self.spec.private
        ws = [self.text("h", "Order options", role="heading"), self.combobox("dd", p["label"], p["options"])]
        ws = apply_layout(ws, self.spec.layout, self.rng)
        ws.append(self.button("submit", "Submit", self.submit))
        return Page("/options", "Options", ws)

    def verify(self):
        ok = self.server.get("submitted") == self.spec.private["opt"]
        return Verification(success=ok, score=float(ok), progress=float(ok),
                            evidence={"submitted": self.server.get("submitted")})


class Menu(App):
    family, category = "menu", "navigation"
    TEMPLATES = ['Choose "{item}" from the Actions menu.', 'Use the Actions menu to select "{item}".',
                 'Open Actions and click "{item}".']

    @classmethod
    def make(cls, rng, split):
        items = rng.sample(MENU_ITEMS, 4)
        view = rng.sample(["Compact", "Comfortable", "Threaded", "Unread first"], 3)
        return {"items": items, "view": view, "item": rng.choice(items)}

    def _pick(self, menu, item):
        self.server["picked"] = (menu, item)
        self.terminal = True

    def _toggle(self, k):
        self.expanded = {k: not self.expanded.get(k, False)}  # opening one closes the other

    def render(self):
        p = self.spec.private
        ws = [self.text("m", "Message: Quarterly report draft", role="heading"),
              self.text("body", "Hi team, please review the attached draft.")]
        for k, name, items in (("act", "Actions", p["items"]), ("view", "View", p["view"])):
            ex = self.expanded.get(k, False)
            ws.append(W(key=k, role="menubutton", name=name, expanded=ex, on_click=lambda k=k: self._toggle(k)))
            if ex:
                ws += [W(key=f"{k}_{i}", role="menuitem", name=it,
                         on_click=lambda name=name, it=it: self._pick(name, it)) for i, it in enumerate(items)]
        return Page("/inbox/42", "Inbox", apply_layout(ws, self.spec.layout, self.rng))

    def verify(self):
        ok = self.server.get("picked") == ("Actions", self.spec.private["item"])
        prog = 1.0 if ok else (0.5 if self.expanded.get("act") else 0.0)
        return Verification(success=ok, score=float(ok), progress=prog, evidence={"picked": self.server.get("picked")})


class SearchLookup(App):
    family, category = "search_lookup", "retrieval"
    TEMPLATES = ['Find the price of "{product}" and finish with the price.',
                 'Search for "{product}" and report its price using finish.',
                 'What does "{product}" cost? Look it up and finish with the answer.']

    @classmethod
    def make(cls, rng, split):
        prods = product_names(rng, 8)
        prices = {p_: price(rng) for p_ in prods}
        target = rng.choice(prods)
        return {"catalog": prices, "product": target, "answer": prices[target], "delay": rng.randint(1, 2)}

    def setup(self):
        self.url = "/search"
        self.server["results"] = None

    def do_search(self):
        q = _norm(self.field_values.get("q", ""))
        cat = self.spec.private["catalog"]
        self.server["results"] = None
        self.server["queries"] = self.server.get("queries", []) + [q]

        def deliver(q=q):
            words = q.split()
            res = [n for n in cat if not words or any(w in _norm(n) for w in words)]
            self.server["results"] = res[:5]

        self.later(self.spec.private["delay"], deliver)

    def render(self):
        cat = self.spec.private["catalog"]
        ws = [self.textbox("q", "Search products", role="searchbox"), self.button("go", "Search", self.do_search)]
        res = self.server.get("results")
        if res is not None:
            if not res:
                ws.append(self.text("none", "No results"))
            for i, n in enumerate(res):
                ws.append(self.text(f"r{i}", f"{n} - {cat[n]}"))
        return Page("/search", "Product search", apply_layout(ws, self.spec.layout, self.rng))

    def verify(self):
        p = self.spec.private
        ans = _norm(self.server.get("answer"))
        ok = bool(self.server.get("finished")) and ans in (_norm(p["answer"]), _norm(p["answer"]).lstrip("$"))
        prog = 0.5 if (self.server.get("results") and p["product"] in self.server["results"]) else 0.0
        return Verification(success=ok, score=float(ok), progress=1.0 if ok else prog,
                            evidence={"answer": self.server.get("answer")})


class FormValidation(App):
    family, category = "form_validation", "recovery"
    TEMPLATES = ['Sign up with name "{name}" and email "{email}".',
                 'Create an account for "{name}" using the email "{email}".',
                 'Register "{name}" with email address "{email}".']

    @classmethod
    def make(cls, rng, split):
        name = rng.choice(vocab(split, PEOPLE_TRAIN, PEOPLE_TEST))
        email = name.split()[0].lower() + str(rng.randint(1, 99)) + "@example.com"
        return {"name": name, "email": email, "flaky": rng.random() < 0.35, "extra": rng.random() < 0.5}

    def setup(self):
        self.url = "/signup"
        self.server["attempts"] = 0

    def submit(self):
        self.server["attempts"] += 1
        name, email = self.field_values.get("name", ""), self.field_values.get("email", "")
        if not name.strip():
            self.server["error"] = "Full name is required"
            return
        if not email.strip():
            self.server["error"] = "Email is required"
            return
        if "@" not in email:
            self.server["error"] = "Enter a valid email address"
            return
        if self.spec.private["flaky"] and not self.server.get("flaked"):
            self.server["flaked"] = True
            self.server["error"] = "Server busy, please try again"
            return
        self.server["error"] = None
        self.server["submitted"] = {"name": name, "email": email}
        self.goto("/welcome")
        self.terminal = True

    def render(self):
        if self.url == "/welcome":
            return Page("/welcome", "Welcome", [self.text("ok", "Account created!", role="heading")])
        ws = [self.textbox("name", "Full name"), self.textbox("email", "Email")]
        if self.spec.private["extra"]:
            ws.append(self.textbox("phone", "Phone (optional)"))
        ws = apply_layout(ws, self.spec.layout, self.rng)
        if self.server.get("error"):
            ws.append(self.text("err", self.server["error"], role="alert"))
        ws.append(self.button("submit", "Create account", self.submit))
        return Page("/signup", "Sign up", ws)

    def verify(self):
        p = self.spec.private
        sub = self.server.get("submitted") or {}
        ok = _norm(sub.get("name")) == _norm(p["name"]) and _norm(sub.get("email")) == _norm(p["email"])
        return Verification(success=ok, score=float(ok), progress=float(ok),
                            evidence={"submitted": sub, "attempts": self.server.get("attempts")})


class Checkout(App):
    family, category = "checkout", "multistep"
    allowed_paths = ("/shop", "/cart", "/checkout")
    TEMPLATES = ['Buy "{product}" and ship it to "{name}" in "{city}".',
                 'Order one "{product}" for "{name}", delivered to "{city}".',
                 'Purchase "{product}"; the recipient is "{name}" in "{city}".']

    @classmethod
    def make(cls, rng, split):
        prods = product_names(rng, rng.randint(3, 5))
        return {"products": {p_: price(rng) for p_ in prods}, "product": rng.choice(prods),
                "name": rng.choice(vocab(split, PEOPLE_TRAIN, PEOPLE_TEST)),
                "city": rng.choice(vocab(split, CITIES_TRAIN, CITIES_TEST))}

    def setup(self):
        self.url = "/shop"
        self.server["cart"] = []
        self.server["orders"] = []

    def add(self, prod):
        self.server["cart"] = self.server["cart"] + [prod]

    def place(self):
        name, city = self.field_values.get("name", ""), self.field_values.get("city", "")
        if not name.strip() or not city.strip():
            self.server["error"] = "Name and city are required"
            return
        self.server["error"] = None
        self.server["orders"].append({"items": list(self.server["cart"]), "name": name, "city": city})
        self.server["cart"] = []
        self.goto("/confirmation")
        self.terminal = True

    def render(self):
        prods = self.spec.private["products"]
        cart = self.server["cart"]
        if self.url == "/shop":
            ws = []
            for i, (n, pr) in enumerate(prods.items()):
                ws.append(self.text(f"p{i}", f"{n} - {pr}"))
                ws.append(self.button(f"add{i}", f"Add {n} to cart", lambda n=n: self.add(n)))
            ws = apply_layout(ws, self.spec.layout, self.rng)
            ws.insert(0, self.link("cart", f"Cart ({len(cart)})", "/cart"))
            return Page("/shop", "Shop", ws)
        if self.url == "/cart":
            ws = [self.text("h", "Your cart", role="heading")]
            ws += [self.text(f"c{i}", n) for i, n in enumerate(cart)] or [self.text("empty", "Cart is empty")]
            ws.append(W(key="co", role="link", name="Checkout", href="/checkout", disabled=not cart))
            ws.append(self.link("back", "Continue shopping", "/shop"))
            return Page("/cart", "Cart", ws)
        if self.url == "/checkout":
            ws = [self.textbox("name", "Full name"), self.textbox("city", "City")]
            ws = apply_layout(ws, self.spec.layout, self.rng)
            if self.server.get("error"):
                ws.append(self.text("err", self.server["error"], role="alert"))
            ws.append(self.button("place", "Place order", self.place))
            return Page("/checkout", "Checkout", ws)
        return Page("/confirmation", "Order confirmed", [self.text("ok", "Thank you for your order!",
                                                                    role="heading")])

    def verify(self):
        p = self.spec.private
        orders = self.server.get("orders", [])
        ok = (len(orders) == 1 and orders[0]["items"] == [p["product"]]
              and _norm(orders[0]["name"]) == _norm(p["name"]) and _norm(orders[0]["city"]) == _norm(p["city"]))
        prog = 0.0
        if p["product"] in self.server.get("cart", []) or orders:
            prog = 0.33
        if self.url in ("/checkout", "/confirmation"):
            prog = 0.66
        return Verification(success=ok, score=float(ok), progress=1.0 if ok else prog, evidence={"orders": orders})


# ---------------------------------------------------------------- held-out families
class DialogConfirm(App):
    family, category = "dialog_confirm", "recovery"
    TEMPLATES = ['Delete the file "{name}".', 'Remove "{name}" from the file list.',
                 'Get rid of the file named "{name}".']

    @classmethod
    def make(cls, rng, split):
        files = [w.lower() + rng.choice([".txt", ".pdf", ".png"])
                 for w in rng.sample(vocab(split, WORDS_TRAIN, WORDS_TEST), rng.randint(3, 5))]
        return {"files": files, "name": rng.choice(files)}

    def setup(self):
        self.url = "/files"
        self.server["deleted"] = []
        self.server["dialog"] = None

    def confirm(self):
        f = self.server["dialog"]
        self.server["deleted"].append(f)
        self.server["dialog"] = None
        self.terminal = True

    def render(self):
        files = [f for f in self.spec.private["files"] if f not in self.server["deleted"]]
        ws = [self.text("h", "Files", role="heading")]
        for i, f in enumerate(files):
            ws.append(self.text(f"f{i}", f))
            ws.append(self.button(f"d{i}", f"Delete {f}",
                                  lambda f=f: self.server.__setitem__("dialog", f)))
        ws = apply_layout(ws, self.spec.layout, self.rng)
        if self.server["dialog"]:
            ws.append(W(key="dlg", role="dialog", name=f"Delete {self.server['dialog']}?"))
            ws.append(self.button("ok", "Confirm", self.confirm))
            ws.append(self.button("cancel", "Cancel", lambda: self.server.__setitem__("dialog", None)))
        return Page("/files", "Files", ws)

    def verify(self):
        ok = self.server["deleted"] == [self.spec.private["name"]]
        return Verification(success=ok, score=float(ok), progress=float(ok),
                            evidence={"deleted": self.server["deleted"]})


class ScrollFind(App):
    family, category = "scroll_find", "partial_observability"
    TEMPLATES = ['Click the "{t}" button.', 'Press the button labeled "{t}".', 'Find and click "{t}".']

    @classmethod
    def make(cls, rng, split):
        labels = rng.sample(_WORDS_ALL, 22)
        tgt_idx = rng.randint(14, 21)
        return {"labels": labels, "t": labels[tgt_idx]}

    def _click(self, lab):
        self.server["clicked"] = lab
        self.terminal = True

    def render(self):
        ws = [self.button(f"b{i}", lab, lambda lab=lab: self._click(lab))
              for i, lab in enumerate(self.spec.private["labels"])]
        return Page("/list", "Long list", ws, scrollable=True)

    def verify(self):
        ok = self.server.get("clicked") == self.spec.private["t"]
        return Verification(success=ok, score=float(ok), progress=float(ok),
                            evidence={"clicked": self.server.get("clicked")})


class ToggleSettings(App):
    family, category = "toggle_settings", "form"
    TEMPLATES = ['Turn on "{on}" and turn off "{off}", then save.',
                 'Enable "{on}", disable "{off}", and save the settings.',
                 'Make sure "{on}" is on and "{off}" is off, then press Save.']

    @classmethod
    def make(cls, rng, split):
        names = rng.sample(SETTINGS, 4)
        init = {n: rng.random() < 0.5 for n in names}
        on, off = rng.sample(names, 2)
        init[on] = False if rng.random() < 0.8 else init[on]
        init[off] = True if rng.random() < 0.8 else init[off]
        return {"names": names, "init": init, "on": on, "off": off}

    def setup(self):
        self.url = "/settings"
        self.server["state"] = dict(self.spec.private["init"])

    def render(self):
        names = self.spec.private["names"]
        ws = [self.text("h", "Settings", role="heading")]
        for i, n in enumerate(names):
            if f"s{i}" not in self.checks:
                self.checks[f"s{i}"] = self.server["state"][n]
            ws.append(self.checkbox(f"s{i}", n, role="switch"))
        ws = apply_layout(ws, self.spec.layout, self.rng)
        ws.append(self.button("save", "Save", self.save))
        return Page("/settings", "Settings", ws)

    def save(self):
        names = self.spec.private["names"]
        self.server["saved"] = {n: self.checks.get(f"s{i}", False) for i, n in enumerate(names)}
        self.terminal = True

    def verify(self):
        p = self.spec.private
        s = self.server.get("saved")
        ok = bool(s) and s[p["on"]] and not s[p["off"]] and all(
            s[n] == p["init"][n] for n in p["names"] if n not in (p["on"], p["off"]))
        return Verification(success=ok, score=float(ok), progress=float(ok), evidence={"saved": s})


class LookupThenFill(App):
    family, category = "lookup_then_fill", "multiapp"
    allowed_paths = ("/orders", "/support")
    TEMPLATES = ['Find the order number for "{customer}" on the Orders page and submit it in the Support form.',
                 'Look up "{customer}"\'s order number, then enter it in the Support form and submit.',
                 'Copy the order number of "{customer}" into the Support form and submit.']

    @classmethod
    def make(cls, rng, split):
        people = rng.sample(vocab(split, PEOPLE_TRAIN, PEOPLE_TEST), 4)
        codes = {p_: "#" + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(5))
                 for p_ in people}
        return {"codes": codes, "customer": rng.choice(people)}

    def setup(self):
        self.url = "/orders"

    def submit(self):
        self.server["submitted"] = self.field_values.get("num", "")
        self.terminal = True

    def render(self):
        codes = self.spec.private["codes"]
        if self.url == "/orders":
            ws = [self.text("h", "Orders", role="heading")]
            ws += [self.text(f"o{i}", f"Order {c} - {n}") for i, (n, c) in enumerate(codes.items())]
            ws = apply_layout(ws, self.spec.layout, self.rng)
            ws.append(self.link("sup", "Support form", "/support"))
            return Page("/orders", "Orders", ws)
        ws = [self.text("h", "Support", role="heading"), self.textbox("num", "Order number"),
              self.button("submit", "Submit", self.submit), self.link("ord", "Orders", "/orders")]
        return Page("/support", "Support", ws)

    def verify(self):
        p = self.spec.private
        ok = _norm(self.server.get("submitted")) == _norm(p["codes"][p["customer"]])
        return Verification(success=ok, score=float(ok), progress=float(ok),
                            evidence={"submitted": self.server.get("submitted")})


FAMILIES: dict[str, type[App]] = {
    c.family: c
    for c in (ClickButton, NavigateLink, FillField, Login, Checkboxes, Dropdown, Menu, SearchLookup,
              FormValidation, Checkout, DialogConfirm, ScrollFind, ToggleSettings, LookupThenFill)
}

TRAIN_FAMILIES = ["click_button", "navigate_link", "fill_field", "login", "checkboxes", "dropdown", "menu",
                  "search_lookup", "form_validation", "checkout"]
HELDOUT_FAMILIES = ["dialog_confirm", "scroll_find", "toggle_settings", "lookup_then_fill"]

MAX_STEPS = {"click_button": 6, "navigate_link": 8, "fill_field": 8, "login": 10, "checkboxes": 10,
             "dropdown": 8, "menu": 6, "search_lookup": 10, "form_validation": 12, "checkout": 20,
             "dialog_confirm": 8, "scroll_find": 10, "toggle_settings": 10, "lookup_then_fill": 12}

DIFFICULTY = {"click_button": 1, "menu": 2, "fill_field": 2, "dropdown": 2, "navigate_link": 2,
              "checkboxes": 3, "login": 3, "search_lookup": 3, "form_validation": 4, "checkout": 5,
              "dialog_confirm": 2, "scroll_find": 2, "toggle_settings": 3, "lookup_then_fill": 4}

Maker = Callable[[random.Random, str], dict]
