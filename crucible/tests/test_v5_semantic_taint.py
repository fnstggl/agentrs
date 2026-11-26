"""The E.5 data-flow audit (`crucible.semantic.taint`) on planted code: it must find creator-declared scores that
reach participant decisions by any of the usual routes, and must not flag binding terms, verified declarations,
third-party signals or content measured through the shared space."""
from __future__ import annotations

import textwrap

from crucible.semantic.taint import Analysis

HEADER = """
from crucible.core import schema as S
from crucible.core.kernel import Module, applier, endpoint
from crucible.population import register_activity, record_decisions
TABLE = "items"
"""

SHOP = HEADER + """
class Shop(Module):
    name = "shop"

    @endpoint("list_item", S.obj({"title": S.s_str(), "quality": S.s_num(), "price": S.s_int(),
                                  "duration_s": S.s_num()}))
    def ep_list(self, ctx, title, quality, price, duration_s):
        measured = self.measure(ctx, title)
        if abs(duration_s - measured) > 1.0:
            raise ValueError("duration must agree with the media")
        ctx.emit("shop.listed", {"seller": ctx.actor, "title": title, "quality": quality, "price": price,
                                 "duration_s": duration_s})

    def measure(self, ctx, title):
        return len(title) * 0.5

    @applier("shop.listed")
    def ap_listed(self, ev):
        p = ev.payload
        self.world.state.put(TABLE, p["title"], {"seller": p["seller"], "quality": p["quality"],
                                                 "price": p["price"], "duration_s": p["duration_s"]})

    @endpoint("buy", S.obj({"item": S.s_str()}))
    def ep_buy(self, ctx, item):
        rec = ctx.state.get(TABLE, item)
        bank_move(ctx, ctx.actor, rec["seller"], rec["price"])

    @endpoint("rate", S.obj({"item": S.s_str(), "stars": S.s_int()}))
    def ep_rate(self, ctx, item, stars):
        rec = ctx.state.get(TABLE, item)
        if rec["seller"] == ctx.actor:
            raise ValueError("you cannot rate your own item")
        ctx.emit("shop.rated", {"by": ctx.actor, "item": item, "stars": stars})

    @applier("shop.rated")
    def ap_rated(self, ev):
        self.world.state.put("ratings", ev.payload["item"] + ev.payload["by"],
                             {"by": ev.payload["by"], "item": ev.payload["item"], "stars": ev.payload["stars"]})


def bank_move(ctx, a, b, cents):
    return ctx.call("bank.transfer", {"from": a, "to": b, "amount": cents})
"""


def run(files: dict[str, str], tmp_path) -> dict:
    for name, text in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text))
    a = Analysis(str(tmp_path))
    a.run()
    return a.report()


def test_direct_read_of_a_declared_score_is_found(tmp_path):
    pop = HEADER + """
def browse(ctx, idx, now):
    rec = ctx.world.state.get(TABLE, "x")
    return rec["quality"] * 2.0

def install(world):
    register_activity(world, "shop", browse, owner="shop")
"""
    r = run({"shop.py": SHOP, "pop.py": pop}, tmp_path)
    flows = [v["flow"] for v in r["violations"]]
    assert any("items.quality" in f for f in flows), r["violations"]
    assert not any("items.price" in f for f in flows)        # binding term (charged by bank.transfer)
    assert not any("duration_s" in f for f in flows)         # verified against a measurement


def test_score_through_helpers_records_and_decisions_is_found(tmp_path):
    pop = HEADER + """
def item_record(world, key):
    return world.state.get(TABLE, key)

def appeal(rec):
    return 0.5 + rec["quality"]

def browse(ctx, idx, now):
    rec = item_record(ctx.world, "x")
    u = appeal(rec)
    record_decisions(ctx, "open", idx, {"open": u > 0.7}, model_id="m", model_version="1", fidelity_id="f")
    return 0.0

def install(world):
    register_activity(world, "shop", browse, owner="shop")
"""
    r = run({"shop.py": SHOP, "pop.py": pop}, tmp_path)
    assert any("items.quality" in v["flow"] and v["sink"] == "record_decisions" for v in r["violations"]), r


def test_binding_verified_third_party_and_space_measurements_are_not_flagged(tmp_path):
    pop = HEADER + """
from crucible.semantic import space

def browse(ctx, idx, now):
    rec = ctx.world.state.get(TABLE, "x")
    stars = [r["stars"] for _, r in ctx.world.state.scan("ratings")]
    feats = space.item_features(ctx, rec["title"])
    u = feats["lm_logprob_per_token"] - rec["price"] / 100.0 + sum(stars) + rec["duration_s"]
    return u

def install(world):
    register_activity(world, "shop", browse, owner="shop")
"""
    r = run({"shop.py": SHOP, "pop.py": pop}, tmp_path)
    assert r["violations"] == [], r["violations"]
    assert "Shop.rate:stars" in r["third_party_parameters"]
    assert "Shop.list_item:duration_s" in r["verified_parameters"]


def test_self_rating_without_an_independence_check_counts(tmp_path):
    shop = SHOP.replace('''        if rec["seller"] == ctx.actor:
            raise ValueError("you cannot rate your own item")
''', "")
    pop = HEADER + """
def browse(ctx, idx, now):
    return sum(r["stars"] for _, r in ctx.world.state.scan("ratings"))

def install(world):
    register_activity(world, "shop", browse, owner="shop")
"""
    r = run({"shop.py": shop, "pop.py": pop}, tmp_path)
    assert any("ratings.stars" in v["flow"] for v in r["violations"]), r


def test_marked_lines_are_reported_not_dropped(tmp_path):
    pop = HEADER + """
def browse(ctx, idx, now):
    rec = ctx.world.state.get(TABLE, "x")
    record_decisions(ctx, "open", idx, {"q": rec["quality"]}, model_id="m", model_version="1", fidelity_id="f")  # taint-ok: planted for the test
    return 0.0

def install(world):
    register_activity(world, "shop", browse, owner="shop")
"""
    r = run({"shop.py": SHOP, "pop.py": pop}, tmp_path)
    assert r["violations"] == [] and r["marked"] and "planted" in r["marked"][0]["reason"]
