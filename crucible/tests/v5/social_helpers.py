"""Shared helpers for the social-internet tests (workstream C1): a world with the whole social internet on the 4.0
services, participants admitted through the operator tool, and a minimal HTML form driver for the browser-mode
pages (TestClient level). Real modules only."""
from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import urljoin

import crucible
from crucible.core.kernel import World, WorldSpec

crucible.load_all_modules()

from crucible.social.modules import social_internet_modules  # noqa: E402


def social_world(seed: int = 1, extra: list[str] | None = None, *, params: dict | None = None,
                 people=("alice", "bob", "carol", "dave"), endowment: int = 100_000, store=None) -> tuple[World, dict]:
    spec = WorldSpec("social-c1", seed, social_internet_modules(extra), dict(params or {}))
    w = World.create(spec, store=store)
    P = {}
    for h in people:
        P[h] = w.system_call("admin.join", {"handle": h, "endowment": endowment, "password": f"pw-{h}"})["principal"]
    return w, P


class Actor:
    """Calls tools as one principal; `ok(...)` asserts success, `fail(...)` asserts a refusal and returns its code."""

    def __init__(self, w: World, principal: str):
        self.w, self.p = w, principal

    def __call__(self, tool: str, **args):
        return self.w.execute(self.p, {"tool": tool, "args": args})

    def ok(self, tool: str, **args):
        r = self(tool, **args)
        assert r.ok, (tool, args, r.error)
        return r.result

    def fail(self, tool: str, **args) -> str:
        r = self(tool, **args)
        assert not r.ok, (tool, args, r.result)
        return r.error["code"]


# ---------------------------------------------------------------------------------------------------- HTML forms
class _Forms(HTMLParser):
    def __init__(self):
        super().__init__()
        self.forms: dict[str, dict] = {}
        self.cur = None
        self.select = None
        self.textarea = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "form":
            self.cur = {"action": a.get("action", ""), "method": (a.get("method") or "get").lower(), "fields": {},
                        "buttons": []}
            if a.get("id"):
                self.forms[a["id"]] = self.cur
        elif self.cur is None:
            return
        elif tag == "input":
            t = (a.get("type") or "text").lower()
            if t == "submit":
                if a.get("name"):
                    self.cur["buttons"].append((a["name"], a.get("value", ""), a.get("value", "")))
            elif t == "checkbox":
                if "checked" in a:
                    self.cur["fields"][a.get("name")] = a.get("value", "on")
            elif a.get("name"):
                self.cur["fields"][a["name"]] = a.get("value", "")
        elif tag == "button" and a.get("name"):
            self.cur["buttons"].append((a["name"], a.get("value", ""), a.get("aria-label") or a.get("value", "")))
        elif tag == "select":
            self.select = a.get("name")
            self.cur["fields"].setdefault(self.select, "")
        elif tag == "option" and self.select is not None:
            if "selected" in a or self.cur["fields"].get(self.select) == "":
                if "selected" in a or not self.cur["fields"][self.select]:
                    self.cur["fields"][self.select] = a.get("value", "")
        elif tag == "textarea":
            self.textarea = a.get("name")
            self.cur["fields"][self.textarea] = ""

    def handle_endtag(self, tag):
        if tag == "form":
            self.cur = None
        elif tag == "select":
            self.select = None
        elif tag == "textarea":
            self.textarea = None

    def handle_data(self, data):
        if self.textarea and self.cur is not None:
            self.cur["fields"][self.textarea] += data


def forms_of(html: str) -> dict[str, dict]:
    p = _Forms()
    p.feed(html)
    return p.forms


def submit(client, page_url: str, form_id: str, values: dict | None = None, button: str | None = None):
    page = client.get(page_url)
    assert page.status_code == 200, (page_url, page.status_code, page.text[:800])
    forms = forms_of(page.text)
    assert form_id in forms, f"no form {form_id!r} on {page_url}: {sorted(forms)}"
    f = forms[form_id]
    data = dict(f["fields"])
    for k, v in (values or {}).items():
        if v is None:
            data.pop(k, None)
        else:
            data[k] = v
    if button is not None:
        b = next(b for b in f["buttons"] if b[2] == button or b[1] == button)
        data[b[0]] = b[1]
    url = urljoin(str(page.url), f["action"])
    if f["method"] == "get":
        return client.get(url, params=data)
    return client.post(url, data=data)


def login(client, handle: str, pw: str, start: str = "http://id.crucible/"):
    r = client.get(start)
    assert "login-form" in r.text
    return submit(client, str(r.url), "login-form", {"handle": handle, "password": pw})


def alerts(html: str) -> list[str]:
    return [re.sub(r"<[^>]+>", "", m).strip() for m in re.findall(r'role="alert"[^>]*>(.*?)</div>', html, re.S)]


def statuses(html: str) -> list[str]:
    return [re.sub(r"<[^>]+>", "", m).strip() for m in re.findall(r'role="status"[^>]*>(.*?)</div>', html, re.S)]


# ---------------------------------------------------------------------------------------------------- criteria
def assert_met(req: str, metrics: dict) -> None:
    """Assert every ACCEPTANCE_MANIFEST criterion of requirement `req` on the measured metrics, with the acceptance
    checker's own rule (a gate test fails when its measurement misses a criterion, not only the later check)."""
    from crucible.acceptance.v5 import manifest as MF
    from crucible.acceptance.v5.checker import criterion_met
    crit = next(e for e in MF.load()["requirements"] if e["id"] == req)["criteria"]
    fails = [msg for c in crit for ok, msg in [criterion_met(c, metrics)] if not ok]
    assert not fails, f"{req}: {fails}"
