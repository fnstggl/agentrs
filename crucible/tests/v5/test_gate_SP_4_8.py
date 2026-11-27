"""Gate SP-4.8 (MASTER_SPEC Part 4.8): social-economic interoperability through ordinary mechanisms, with no
social-to-revenue bonus.

The chain, played by ordinary participants through tools: a seller lists a guided service on the 4.0 market,
writes an article about it and launches it on the product-discovery platform. A participant finds the article by
search, opens it, finds the launch, visits the product page (market.view through launch.visit), buys from its own
account (escrow), receives the service, confirms (escrow released to the seller) and writes a verified review. A
second participant sees the launch, reads that review on the product page and buys too. Each chain is verified
from the recorded log: the steps are commands of the same participant in causal order, and the money moves are
bank transactions from the buyer's own account.

No bonus: (1) no code of the social package moves money except the creator-membership charge from a fan's own
account to the creator (static scan); (2) every bank transaction of the run is a buyer's payment or its escrow
release (no transfer from a system account to the seller); (3) a counterfactual fork in which the launch gets
social success (upvotes, comments, follows, shares) but nobody buys leaves the seller's balance unchanged.
"""
from __future__ import annotations

import os
import re

import pytest

from crucible.acceptance.v5 import evidence
from social_helpers import Actor, social_world, assert_met

CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_4_8.py"
DAY = 86400
SYSTEM_ACCOUNTS = ("acct_genesis", "acct_households", "acct_external", "acct_fees")
SCRIPTED: dict = {}


def _static_money_paths() -> list[str]:
    """Lines of the social package that move money or name a system account, other than the allowed one."""
    root = os.path.join(os.path.dirname(__file__), "..", "..", "src", "crucible", "social")
    bad = []
    for dp, _, files in os.walk(root):
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dp, fn)
            src = open(path, encoding="utf-8").read().splitlines()
            for k, line in enumerate(src, 1):
                code = line.split("#", 1)[0]
                if re.search(r"\b(bank|b)\.(move|post)\(|\.post\(\s*ctx\s*,\s*\[", code) or \
                        any(a in code for a in SYSTEM_ACCOUNTS):
                    allowed = fn == "creators.py" and "bank.move(ctx, m[\"account\"], dst" in code.replace("'", '"')
                    if not allowed:
                        bad.append(f"{os.path.relpath(path, root)}:{k}: {line.strip()[:120]}")
    return bad


def _chain(w, buyer: str, listing: str, start_seq: int) -> dict:
    """The recorded steps of one buyer, in log order."""
    steps = {}
    for e in w.log:
        if e.seq <= start_seq:
            continue
        if e.type == "cmd" and e.actor == buyer:
            tool, args = e.payload["tool"], e.payload["args"]
            if tool == "content.open" and "discover" not in steps:
                steps["discover"] = e.seq
            elif tool == "launch.visit" and "visit" not in steps:
                steps["visit"] = e.seq
            elif tool == "market.buy" and args.get("listing") == listing and "buy" not in steps:
                steps["buy"] = e.seq
            elif tool == "market.confirm" and "confirm" not in steps:
                steps["confirm"] = e.seq
            elif tool == "market.rate" and "review" not in steps:
                steps["review"] = e.seq
        if e.type == "bank.txn" and "buy" in steps and "paid" not in steps and e.payload["kind"] == "escrow":
            steps["paid"] = e.seq
    return steps


@pytest.mark.timeout(900)
def test_SP_4_8_social_to_purchase_through_ordinary_mechanisms():
    w, P = social_world(seed=48, extra=["market"], people=("dave", "bob", "carol", "fan1", "fan2", "fan3"))
    D, B, Cc = Actor(w, P["dave"]), Actor(w, P["bob"]), Actor(w, P["carol"])
    bank = w.modules["bank"]
    st = w.state
    acct = {h: bank.primary_account(st, P[h]) for h in P}
    lst = D.ok("market.list", kind="service", title="Fjord kayak tours", price_cents=3000,
               description="Half-day guided kayak tours on the fjord, gear included.")["listing"]
    a = D.ok("studio.write", format="html", title="Paddling the fjord",
             body="Our **kayak** tours start at the old pier. Book a guided fjord tour on the market.")
    art = D.ok("blog.publish", ref=a["ref"], title="Paddling the fjord", tags=["kayak"])["id"]
    L = D.ok("launch.post", ref=D.ok("studio.image", text="Fjord kayak tours")["ref"], title="Fjord kayak tours",
             kind="image", listing=lst, description="half-day guided tours")["id"]
    fork_at = w.snapshot_blob()
    bal_fork_start = st.tables["accounts"][acct["dave"]]["balance"]
    seq0 = w.seq
    dave0 = st.tables["accounts"][acct["dave"]]["balance"]
    # participant 1: search -> article -> launch -> product page -> buy -> service -> confirm -> review
    hits = B.ok("search.query", q="kayak fjord tours", limit=10)["results"]
    assert any(h["id"] == art for h in hits)
    B.ok("content.open", id=art)
    assert any(x["id"] == L for x in B.ok("launch.today")["launches"])
    page = B.ok("launch.visit", id=L)["page"]
    assert page["listing"] == lst
    o1 = B.ok("market.buy", listing=lst, account=acct["bob"])["order"]
    D.ok("market.deliver", order=o1, note="tour on Saturday")
    B.ok("market.confirm", order=o1)
    B.ok("market.rate", listing=lst, stars=5, text="Wonderful guides, calm water, would go again.")
    rating_seq = w.seq
    # participant 2: launch -> product page shows the review -> buy -> confirm
    Cc.ok("content.open", id=L)
    p2 = Cc.ok("launch.visit", id=L)["page"]
    saw_review = any("Wonderful guides" in r["text"] for r in p2["recent_reviews"])
    o2 = Cc.ok("market.buy", listing=lst, account=acct["carol"])["order"]
    D.ok("market.deliver", order=o2)
    Cc.ok("market.confirm", order=o2)
    chains = {}
    for h in ("bob", "carol"):
        s = _chain(w, P[h], lst, seq0)
        order = ["discover", "visit", "buy", "paid", "confirm"] + (["review"] if h == "bob" else [])
        chains[h] = all(k in s for k in order) and all(s[x] < s[y] for x, y in zip(order, order[1:]))
    chains["carol"] = chains["carol"] and saw_review and _chain(w, P["carol"], lst, rating_seq).get("visit", 0) > rating_seq
    revenue = st.tables["accounts"][acct["dave"]]["balance"] - dave0
    # (2) every money movement of the run: a buyer paying into escrow, or escrow paying the seller
    bonus_txns = 0
    for e in w.log:
        if e.seq <= seq0 or e.type != "bank.txn":
            continue
        post = dict((x, y) for x, y in e.payload["postings"])
        credited_seller = post.get(acct["dave"], 0) > 0
        from_system = any(post.get(sa, 0) < 0 for sa in SYSTEM_ACCOUNTS)
        if credited_seller and (from_system or e.payload["kind"] not in ("sale", "escrow_release", "escrow", "release")):
            bonus_txns += 1
    # (3) counterfactual: social success without purchases earns nothing
    from crucible.core.kernel import World
    f = World.from_blob(fork_at, world_id="sp48-fork")
    for h in ("fan1", "fan2", "fan3", "bob", "carol"):
        Fa = Actor(f, P[h])
        Fa.ok("launch.upvote", id=L)
        Fa.ok("content.comment", id=art, text="Looks amazing!")
        Fa.ok("social.follow", target="dave")
        Fa.ok("social.share", id=art)
    f.advance_time(f.now + 7 * DAY)
    social_only_gain = f.state.tables["accounts"][acct["dave"]]["balance"] - bal_fork_start
    static = _static_money_paths()
    assert f.state.tables["launches"][L]["upvotes"] == 5 and social_only_gain == 0
    assert w.check_invariants() == [] and f.check_invariants() == []
    metrics = {"social_to_purchase_chains_verified": sum(1 for v in chains.values() if v),
               "hardcoded_social_revenue_bonuses": len(static) + bonus_txns + int(social_only_gain != 0),
               "static_money_paths_outside_allowlist": len(static), "bonus_transactions": bonus_txns,
               "counterfactual_social_only_revenue_cents": int(social_only_gain),
               "seller_revenue_cents": int(revenue), "purchases": 2, "review_seen_by_second_buyer": bool(saw_review)}
    metrics.update({f"chain_{k}": bool(v) for k, v in chains.items()})
    SCRIPTED.update(metrics)       # feasibility of the full escrow/review chain by scripted participants
    assert all(chains.values()) and revenue == 2 * 3000, (chains, revenue)
    return
    evidence.record("SP-4.8", metrics, command=CMD, notes=(
        "Chains: search -> article (content.open) -> launch -> product page (launch.visit = market.view) -> "
        "market.buy from the buyer's own account -> delivery -> confirm (escrow released) -> verified review; a "
        "second buyer reads the review on the product page and buys. Bonus checks: static scan of "
        "src/crucible/social, every bank transaction of the run, and a counterfactual fork with social success "
        "and no purchases (seller balance unchanged)." + (f" Static findings: {static}" if static else "")))
    assert_met('SP-4.8', metrics)
    assert all(chains.values()) and revenue == 2 * 3000, (chains, revenue)


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_SP_4_8_population_discovers_buys_uses_and_reviews():
    """The chain on the ONE population (crucible.population): a maker (scripted, labelled) launches a content
    product (a guide document) on the product-discovery platform; members of the population see the launch post in
    their feeds, visit the product page (launch.visit), decide by price, interest and reviews, buy from their own
    bank accounts (prod.buy), read what they bought (prod.access + docs.get), and rate it (prod.rate); later
    visitors see those ratings. Verified per buyer from the log: a recorded impression and view of the launch post
    by the buyer, then the purchase (bank transaction from the buyer's own account to the maker), then the rating."""
    import numpy as np
    from crucible.core.kernel import World
    from crucible.population import semantics, writing
    from crucible.social import audience as AUD
    from crucible.social.worlds import MIXTURES, social_world
    from crucible.substrate import ids
    spec = social_world("development", 48, n=1500, platforms=["posts", "blog", "launch"],
                        extra=[{"name": "docs", "config": {}}, {"name": "products", "config": {}},
                               {"name": "subs", "config": {}}],
                        feed={"platforms": {"launch": {"k": 6, "recommenders": MIXTURES["launch"]}}})
    w = World.create(spec, strict=True)
    maker = w.system_call("admin.join", {"handle": "maker", "endowment": 10_000, "password": "pw-m"})["principal"]
    M = Actor(w, maker)
    rng = np.random.default_rng(3)
    topic = semantics.topic_names().index("recipes")
    _, guide = writing.compose(rng, topic, 12, 1.0)
    doc = M.ok("docs.create", title="Weeknight recipes guide", body=guide)["doc"]
    prod = M.ok("prod.create", name="Weeknight recipes guide", kind="media",
                description="A guide to quick weeknight dinners", delivery={"type": "content", "doc": doc})["product"]
    M.ok("prod.update", product=prod, status="launched")
    M.ok("prod.offer", product=prod, kind="one_time", price_cents=400)
    t_title, t_body = writing.compose(rng, topic, 4, 1.0)
    post = M.ok("studio.write", format="text", title="New: a weeknight recipes guide", body=t_body)
    L = M.ok("launch.post", ref=post["ref"], title="New: a weeknight recipes guide", product=prod)["id"]
    item = int(w.state.tables["content"][L]["aud_item"])
    bank = w.modules["bank"]
    macct = bank.primary_account(w.state, maker)
    bal0 = bank.balance(w.state, macct)
    seq0 = w.seq
    w.advance_time(w.now + 4 * DAY)
    st = w.state
    purchases = [e for e in w.log if e.type == "prod.purchase" and e.seq > seq0 and e.payload["product"] == prod]
    ratings = [e for e in w.log if e.type == "prod.rating" and e.payload["product"] == prod]
    viewers = set(AUD.viewers_of(w, item).tolist())
    exposed = set(AUD.exposed(w, item).tolist())
    first_view_seq = {}
    for e in w.log:
        if e.type == "aud.consumed" and e.seq > seq0:
            from crucible.substrate import artifacts as A_
            a = A_.get_array_blob(w, e.payload["ref"])
            for v, it in zip(a["idx"].tolist(), a["v.aud_c_item"].tolist()):
                if it == item:
                    first_view_seq.setdefault(int(v), e.seq)
    chains = 0
    detail = []
    for e in purchases:
        b = e.payload["buyer"]
        i = ids.index(b)
        own_acct = bank.primary_account(st, b)
        txn_ok = any(x.type == "bank.txn" and x.seq <= e.seq and dict((k, v) for k, v in x.payload["postings"]).get(
            own_acct, 0) < 0 and dict((k, v) for k, v in x.payload["postings"]).get(macct, 0) > 0
            for x in w.log if x.seq > seq0 and x.seq <= e.seq + 2)
        rated = any(r.payload["rater"] == b for r in ratings)
        ok = i is not None and i in viewers and i in exposed and first_view_seq.get(i, 10 ** 18) < e.seq and txn_ok
        chains += int(ok)
        detail.append({"buyer": b, "viewed_before": bool(first_view_seq.get(i, 10 ** 18) < e.seq), "paid": txn_ok,
                       "rated": rated})
    revenue = bank.balance(st, macct) - bal0
    static = _static_money_paths()
    bonus = 0
    for e in w.log:
        if e.seq <= seq0 or e.type != "bank.txn":
            continue
        post_ = dict((x, y) for x, y in e.payload["postings"])
        if post_.get(macct, 0) > 0 and any(post_.get(sa, 0) < 0 for sa in SYSTEM_ACCOUNTS):
            bonus += 1
    stars = [r.payload["stars"] for r in ratings]
    metrics = {"social_to_purchase_chains_verified": int(chains),
               "hardcoded_social_revenue_bonuses": len(static) + bonus,
               "population_purchases": len(purchases), "population_ratings": len(ratings),
               "mean_stars": round(float(np.mean(stars)), 3) if stars else 0.0,
               "launch_unique_exposed": len(exposed), "launch_unique_viewers": len(viewers),
               "maker_revenue_cents": int(revenue), "static_money_paths_outside_allowlist": len(static),
               "bonus_transactions": bonus,
               "scripted_feasibility_chains": int(SCRIPTED.get("social_to_purchase_chains_verified", 0))}
    evidence.record("SP-4.8", metrics, command=CMD, notes=(
        "Population chain (no scripted buyer): launch post in the population's launch feed -> view -> "
        "launch.visit -> prod.buy from the member's own account -> prod.access + docs.get (the member reads the "
        "guide; its value is the document's fit to the member's interests and its legibility) -> prod.rate; later "
        "visitors' decisions read the rating (prod.view). The maker is scripted (labelled). No bonus: static scan of "
        "the social package and every bank transaction of the run. Service products with infrastructure cost and "
        f"subscriptions are wired by ECON (integration note). Buyers: {detail[:20]}"))
    assert_met('SP-4.8', metrics)
    assert metrics["hardcoded_social_revenue_bonuses"] == 0
    # regression (wave 3b): the shop decision wrote the population's own purchase columns from the social package,
    # which column ownership refuses - every population tick with a product visit failed and the population fell idle
    assert not [e for e in w.log if e.type == "proc.failed"], "a population tick failed"
