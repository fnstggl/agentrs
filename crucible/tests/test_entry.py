"""Market entry by founders (module `entry`): vacated and badly served markets are entered through public tools,
saturated ones are not, bad entrants fail and lose their capital, money is conserved, and everything replays."""
import numpy as np
import pytest

import crucible

crucible.load_all_modules()
from crucible.actors import entry as E  # noqa: E402,F401  (importing registers the module)
from crucible.core.kernel import World  # noqa: E402
from crucible.core.errors import Forbidden  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400
GOOD = {"risk": 0.5, "competence": 0.8, "informed": 0.9, "patience": 0.5}


def _firm(i, need, tier, price, cash=3_000_000):
    stem = need.split(".")[1][:6]
    return {"handle": f"{stem}co{i}", "name": f"{stem.title()}Co {i}", "need": need, "tier": tier, "price": price,
            "host": f"{stem}{i}", "provider": "o:cloud1", "cores": 1, "cash": cash,
            "policy": {"price_step": 0.0, "outsource": False}}


def _world(firms, needs, entry_cfg, seed=1, n=600, params=None, civ=None, workers=0):
    """A small legacy-style world (NPC firms) with an explicit industry structure."""
    ov = {"needs": needs, "n_consumers": n, "workers": workers, "bounties": 0,
          "module_config": {"civ": {"firms": firms, "bounties": [], **(civ or {})}, "entry": entry_cfg,
                            "consumers": {"k_samples": 2}},
          "params": dict(params or {})}
    return World.create(generate({"family": "tiny", "overrides": ov, "extra_modules": ["entry"]}, seed))


def _served(w, need, day):
    st = w.state
    hosts = {l["host"] for _, l in st.scan("listings") if str(l["need"]) == need and l["host"]}
    return sum(r["ok"] for k, r in st.scan("svc_days") if r["host"] in hosts and k.endswith(f"|{day}"))


def _customers(w, need=None):
    """host -> customers it currently has (hidden population state, read only to check outcomes)."""
    st = w.state
    out = {}
    for lid, o in st.scan("cons_offers"):
        if need is None or str(o["need"]) == need:
            host = st.tables["listings"][lid]["host"]
            out[host] = out.get(host, 0) + int((st.arrays["c_prov"] == o["slot"]).sum())
    return out


def _truth(w, need):
    """Hidden ground truth, used only to judge the founders' estimates: consumers who have the need, and how many
    of them have no provider."""
    st = w.state
    mask = st.arrays["c_need"] == [str(x) for x in st.meta_get("cons_needs")].index(need)
    if "c_active" in st.arrays:
        mask = mask & st.arrays["c_active"]
    return int(mask.sum()), int((mask & (st.arrays["c_prov"] < 0)).sum())


def _sound(w):
    bank = w.modules["bank"]
    assert w.check_invariants() == []
    assert bank.check_ledger_consistency(w) == []
    assert sum(a["balance"] for a in w.state.tables["accounts"].values()) == 0
    assert not [e.payload for e in w.log if e.type == "proc.failed"]


def _books_close(w):
    """Every founder's cash is exactly its genesis savings minus research, minus what it put into companies, plus
    what came back: nothing was created for founders and nothing vanished."""
    st = w.state
    for fid, f in st.scan("founders"):
        vs = list(f["ventures"]) + ([f["venture"]] if f["venture"] else [])
        cash = st.tables["accounts"][f["account"]]["balance"]
        assert cash == f["capital0"] - f["stats"]["research_cents"] - sum(v["invested"] for v in vs) \
            + sum(v.get("recovered", 0) for v in vs), fid
        endow = [st.tables["txns"][e["txn"]] for e in st.find("acct_txns", "acct", f["account"])
                 if st.tables["txns"][e["txn"]]["kind"] == "endowment"]
        assert sum(d for t in endow for a, d in t["postings"] if a == f["account"]) == f["capital0"]


def _vacate(w, need):
    """The path of tests/test_firm_recovery.py: the firm's instance is terminated and it cannot pay for a new one,
    so its own logic winds the company up."""
    st, bank, cloud = w.state, w.modules["bank"], w.modules["cloud"]
    gone = []
    for lid, l in list(st.scan("listings")):
        if str(l["need"]) != need or l["status"] != "active":
            continue
        acct = bank.primary_account(st, l["seller"])
        bal = bank.balance(st, acct)
        if bal > 0:
            w._run_system("test-drain", lambda ctx, a=acct, b=bal: bank.move(ctx, a, "acct_genesis", b, "test", "drain"))
        inst = st.tables["deployments"][st.tables["hosts"][l["host"]]["deployment"]]["instance"]
        w._run_system("test-terminate", lambda ctx, i=inst: cloud._terminate(ctx, i, "test"))
        gone.append(l["seller"])
    return gone


# ---------------------------------------------------------------------- re-entry into a vacated market
def test_a_vacated_need_is_served_again_only_with_entry(acceptance):
    """Matched forks of the same world after its only seller of one need was removed (the firm-recovery path):
    with founders active the need is served again within 7 days; with entry paused it stays dead."""
    acceptance.claim("After the only seller of a need is removed, forks with founders active serve the need again "
                     "within 7 simulated days; matched forks with entry paused never do (majority of 3 seeds).")
    need, t0, horizon, rows = "text.extract", 3, 7, []
    for seed in (1, 2, 3):
        # measured compute cost: this test compares outcomes, it does not need bit-identical forks, and the
        # deterministic line-counting model is several times slower
        ov = {"n_consumers": 500, "needs": ["sort.records", need], "firms_per_need": 1, "workers": 4,
              "module_config": {"entry": {"first_scan_day": t0 + 1.0, "scan_interval_days": 2.0}},
              "params": {"cost_model": "measured"}}
        w = World.create(generate({"family": "open_digital", "overrides": ov, "extra_modules": ["entry"]}, seed))
        worlds = [w]
        try:
            w.advance_time(t0 * DAY)
            assert _served(w, need, t0 - 1) > 0
            assert not any(f["stats"]["scans"] for _, f in w.state.scan("founders"))   # nobody has looked yet
            gone = _vacate(w, need)
            assert len(gone) == 1
            blob = w.snapshot_blob()
            a = World.from_blob(blob, world_id=f"entry-on-{seed}", fork=True)
            b = World.from_blob(blob, world_id=f"entry-off-{seed}", fork=True)
            worlds += [a, b]
            b.system_call("admin.entry_pause", {"paused": True})
            for x in (a, b):
                x.advance_time((t0 + horizon) * DAY)
                assert x.state.tables["companies"][gone[0]]["status"] == "dissolved"      # the incumbent wound up
                _sound(x)
            _books_close(a)
            last = t0 + horizon - 1
            first = next((d for d in range(t0 + 1, t0 + horizon) if _served(a, need, d) > 0), None)
            rows.append({"seed": seed, "with": _served(a, need, last), "without": _served(b, need, last),
                         "before": _served(w, need, t0 - 1), "days_to_service": None if first is None else first - t0,
                         "entries": E.entry_stats(a)["entries"], "entries_paused": E.entry_stats(b)["entries"]})
            assert E.entry_stats(b)["entries"] == 0 and _served(b, need, last) == 0
            assert World.replay(a.spec, [e.to_dict() for e in a.log], base_blob=blob).digest() == a.digest()
        finally:
            for x in worlds:
                x.close()
    acceptance.measure(rows=rows, horizon_days=horizon)
    print("MEASURED vacated-need matched forks:", rows)
    ok = [r for r in rows if r["with"] > 0 and r["without"] == 0]
    assert len(ok) >= 2, rows
    assert all(r["without"] == 0 for r in rows), rows


def test_entry_pause_is_an_operator_tool_only():
    w = _world([_firm(1, "sort.records", 0, 1)], ["sort.records"], {"founder_specs": [dict(GOOD, capital=500_000)]})
    try:
        r = w.execute("p:founder-00", {"tool": "admin.entry_pause", "args": {"paused": True}})
        assert not r.ok and r.error["code"] == Forbidden.code
        assert "admin.entry_pause" not in {t["name"] for t in w.available_tools("p:founder-00")}
        assert not any(t["name"].startswith("entry.") for t in w.available_tools("p:founder-00"))
    finally:
        w.close()


# ---------------------------------------------------------------------- which markets attract entry
def test_an_overpriced_low_quality_monopoly_attracts_an_entrant(acceptance):
    need = "route.shortest"            # reference tier 0 scores about 0.71; base price 3 cents
    w = _world([_firm(1, need, 0, 11)], [need],
               {"founder_specs": [dict(GOOD, capital=900_000) for _ in range(2)], "first_scan_day": 6.0,
                "scan_interval_days": 1.0})
    try:
        w.advance_time(6 * DAY + 8 * 3600 - 1)                      # just before the first founder looks
        n_true, unserved_true = _truth(w, need)
        w.advance_time(11 * DAY)
        st = w.state
        stats = E.entry_stats(w)
        assert stats["entries"] >= 1 and stats["launched"] >= 1 and stats["surveys"] >= 1
        first = min((e for _, f in st.scan("founders") for e in f["estimates"]), key=lambda e: e["t"])
        assert first["survey"] and first["enter"] and first["t"] == 6 * DAY + 8 * 3600
        # the estimate came from public signals; against hidden truth the market size is within the survey's error
        assert abs(first["market_est"] - n_true) <= 0.2 * n_true
        assert abs(first["unserved_est"] - unserved_true) <= 0.35 * max(unserved_true, 50)
        entrants = [l for _, l in st.scan("listings") if l["seller"] in E.entrant_orgs(w)]
        assert entrants and all(l["price"] < 11 for l in entrants)          # priced from what customers would pay
        assert sum(_served(w, need, d) for d in (9, 10)) > 0
        shares = _customers(w, need)
        assert sum(v for h, v in shares.items() if h != "shorte1") > 0
        # every founder paid for its research out of its own account
        fees = [t for t in st.tables["txns"].values() if t["kind"] == "fee" and t["memo"].startswith("market survey")]
        assert len(fees) == stats["surveys"] and all(t["by"].startswith("p:founder-") for t in fees)
        _sound(w)
        _books_close(w)
        acceptance.claim("A monopoly charging about twice the median willingness to pay with the lowest reference "
                         "tier is entered by a well-informed founder within 5 days of its first look.")
        acceptance.measure(entries=stats["entries"], market_true=n_true, market_est=first["market_est"],
                           unserved_true=unserved_true, unserved_est=first["unserved_est"],
                           entrant_prices=[l["price"] for l in entrants], shares=shares)
        print("MEASURED monopoly:", acceptance.rec["measured"])
    finally:
        w.close()


def test_a_saturated_cheap_high_quality_market_is_researched_and_not_entered(acceptance):
    need = "route.shortest"
    w = _world([_firm(i, need, 1, 2) for i in (1, 2, 3)], [need],
               {"founder_specs": [dict(GOOD, capital=900_000) for _ in range(2)], "first_scan_day": 7.0,
                "scan_interval_days": 1.0, "screen_min": 0.0, "resurvey_days": 2})
    try:
        w.advance_time(13 * DAY)
        st = w.state
        stats = E.entry_stats(w)
        ests = [e for _, f in st.scan("founders") for e in f["estimates"]]
        assert stats["surveys"] >= 2 and len(ests) >= 2                 # they looked properly ...
        assert stats["entries"] == 0                                    # ... and stayed out
        assert all(not e["enter"] and e["monthly"] < e["hurdle"] * e["at_risk"] for e in ests)
        assert not E.entrant_orgs(w)
        _sound(w)
        _books_close(w)
        acceptance.claim("Three best-tier sellers at two thirds of the reference price: informed founders pay for "
                         "the survey, estimate a loss and do not enter.")
        acceptance.measure(surveys=stats["surveys"], best_monthly_estimate=max(e["monthly"] for e in ests),
                           hurdle_amounts=[round(e["hurdle"] * e["at_risk"]) for e in ests])
        print("MEASURED saturated:", acceptance.rec["measured"])
    finally:
        w.close()


def test_without_a_free_signal_the_default_screen_does_not_even_pay_for_research():
    need = "route.shortest"
    w = _world([_firm(i, need, 1, 2) for i in (1, 2, 3)], [need],
               {"founder_specs": [dict(GOOD, capital=900_000)], "first_scan_day": 7.0, "scan_interval_days": 1.0})
    try:
        w.advance_time(10 * DAY)
        stats = E.entry_stats(w)
        assert stats["scans"] >= 2 and stats["surveys"] == 0 and stats["entries"] == 0
        f = w.state.tables["founders"]["p:founder-00"]
        assert w.state.tables["accounts"][f["account"]]["balance"] == f["capital0"]
    finally:
        w.close()


# ---------------------------------------------------------------------- entrants can fail
def test_a_badly_informed_low_competence_founder_enters_fails_and_loses_its_capital(acceptance):
    """One best-tier, cheap, well-liked monopolist. A founder who does no research, believes the market is huge,
    cannot test its own code and has thin savings enters anyway, wins almost nobody, and winds up at a loss."""
    need = "route.shortest"
    bad = {"risk": 0.9, "competence": 0.25, "informed": 0.05, "patience": 0.0, "capital": 130_000}
    w = _world([_firm(1, need, 1, 2)], [need],
               {"founder_specs": [bad], "first_scan_day": 8.0, "scan_interval_days": 1.0, "prior_customers": 30_000,
                "max_ventures": 1, "noise_sigma": 0.0})      # its error is the wrong belief, not an unlucky draw
    try:
        st = w.state
        w.advance_time(8 * DAY)
        n_true, unserved_true = _truth(w, need)
        peak = 0
        for d in range(9, 24):
            w.advance_time(d * DAY)
            f = st.tables["founders"]["p:founder-00"]
            if f["venture"] and f["venture"].get("host"):
                peak = max(peak, _customers(w, need).get(f["venture"]["host"], 0))
            if f["stats"]["exits"]:
                break
        f = st.tables["founders"]["p:founder-00"]
        assert f["stats"]["entries"] == 1 and f["stats"]["surveys"] == 0
        est = f["estimates"][-1]
        assert est["enter"] and not est["survey"]
        assert f["stats"]["exits"] == 1 and f["stats"]["failed"] == 1 and f["phase"] == "retired", f["venture"]
        v = f["ventures"][0]
        assert v["tier"] == 0 and v["route"] == "write" and v["launched"] is not None
        assert v["outcome"] == "failed" and v["recovered"] < v["invested"]
        assert st.tables["companies"][v["org"]]["status"] in ("dissolved", "bankrupt")
        cash = st.tables["accounts"][f["account"]]["balance"]
        lost = f["capital0"] - cash
        assert lost == v["invested"] - v["recovered"] and lost >= 50_000        # at least the registration fee
        # the belief was far from the (hidden) truth: a market fifty times its real size, a profit where
        # there was a loss
        assert est["market_est"] == 30_000 and n_true <= 600 and est["unserved_est"] > 20 * unserved_true
        assert est["customers_est"] > 2 * max(peak, 1) and est["monthly"] > 0 > v["recovered"] - v["invested"]
        # under-capitalised: it asked the existing lender, whose own rule refuses a company without revenue
        assert f["stats"]["loans_asked"] == 1 and f["stats"]["loans_granted"] == 0
        assert v["loans"] == [{"amount": v["loans"][0]["amount"], "status": "refused"}]
        # the entrant's service really stopped: no live deployment, listing closed
        assert st.tables["listings"][v["listing"]]["status"] == "closed"
        assert w.modules["cloud"].live_deployment(st, v["host"]) is None
        _sound(w)
        _books_close(w)
        acceptance.claim("A founder with no research, a wrong prior, tier-0 code and thin savings enters a "
                         "well-served market, fails and exits with a loss.")
        acceptance.measure(believed_customers=est["customers_est"], peak_customers=int(peak), market_true=n_true,
                           unserved_true=unserved_true, believed_unserved=est["unserved_est"], invested=v["invested"],
                           recovered=v["recovered"], days_alive=round((v["t_end"] - v["t"]) / DAY, 1),
                           reason=v["reason"])
        print("MEASURED bad founder:", acceptance.rec["measured"])
    finally:
        w.close()


# ---------------------------------------------------------------------- brand-new market, determinism
def _new_market_world(seed=3):
    """Customers of `text.extract` exist from the first day but nobody serves them."""
    return _world([_firm(1, "sort.records", 0, 1)], ["sort.records", "text.extract"],
                  {"founder_specs": [dict(GOOD, capital=700_000), dict(GOOD, capital=600_000, competence=0.45)],
                   "first_scan_day": 1.0, "scan_interval_days": 1.0}, seed=seed, n=500,
                  params={"cost_model": "deterministic"})


def test_a_market_nobody_serves_is_entered_and_the_run_is_deterministic():
    w = _new_market_world()
    worlds = [w]
    try:
        w.advance_time(2 * DAY)
        x, y = w.fork("fx"), w.fork("fy")
        worlds += [x, y]
        for z in (w, x, y):
            z.advance_time(4 * DAY)
        st = w.state
        stats = E.entry_stats(w)
        assert stats["launched"] >= 1 and _served(w, "text.extract", 3) > 0
        tiers = sorted(v["tier"] for v in stats["ventures"])
        assert set(tiers) <= {0, 1} and (1 in tiers)        # competence 0.8 writes the better reference tier
        v = next(r for r in stats["ventures"] if r["tier"] == 1)
        repo = st.tables["repos"][f"{v['org'].split(':', 1)[1]}/svc"]
        assert not repo.get("trusted")                      # written through code.commit: it only ever runs sandboxed
        dep = w.modules["cloud"].live_deployment(st, st.tables["listings"][
            next(l for l, r in st.scan("listings") if r["seller"] == v["org"])]["host"])
        assert dep is not None and dep["trusted"] is False
        _sound(w)
        _books_close(w)
        # replay of the log, regeneration through the handlers, and identical forks
        evs = [e.to_dict() for e in w.log]
        assert World.replay(w.spec, evs).digest() == w.digest()
        g = World.regenerate(w.spec, evs)
        worlds.append(g)
        assert [e.hash for e in g.log] == [e.hash for e in w.log] and g.digest() == w.digest()
        # two forks advanced identically stay identical. (A fork is not compared with its parent here: a fork
        # starts fresh sandbox workers, and under the deterministic cost model the first request a fresh worker
        # serves executes more lines than a warm one - a property of the sandbox, see the WP3b report.)
        assert x.digest() == y.digest()
        assert [e.hash for e in x.log] == [e.hash for e in y.log] and len(x.log) > 500
        assert E.entry_stats(x) == E.entry_stats(y)
    finally:
        for z in worlds:
            z.close()


def test_founders_are_created_with_their_own_savings_and_do_nothing_when_the_module_is_absent():
    base = {"family": "tiny", "overrides": {"n_consumers": 200}}
    w0 = World.create(generate(base, 5))
    w1 = World.create(generate(dict(base, extra_modules=["entry"]), 5))
    try:
        assert "founders" not in w0.state.tables and "entry.scan" not in w0.state.tables["sched"]
        fs = dict(w1.state.scan("founders"))
        assert len(fs) == E.DEFAULTS["founders"]
        lo, hi = E.DEFAULTS["capital_cents"]
        for fid, f in fs.items():
            assert lo <= f["capital0"] <= hi
            assert w1.state.tables["accounts"][f["account"]]["balance"] == f["capital0"]
            assert w1.state.tables["persons"][fid]["kind"] == "npc"
            for k in ("risk", "competence", "informed", "patience"):
                assert E.DEFAULTS[k][0] <= f["traits"][k] <= E.DEFAULTS[k][1]
        assert len({tuple(sorted(f["traits"].items())) for f in fs.values()}) == len(fs)    # heterogeneous
        # the endowment comes from the genesis account like every other person's savings
        assert -w1.state.tables["accounts"]["acct_genesis"]["balance"] == \
            -w0.state.tables["accounts"]["acct_genesis"]["balance"] + sum(f["capital0"] for f in fs.values())
        assert E.entry_stats(w0)["founders"] == 0
    finally:
        w0.close()
        w1.close()


# ---------------------------------------------------------------------- implementation routes and finance
def test_a_founder_who_cannot_code_hires_a_developer_from_the_labour_pool():
    need = "text.extract"
    w = _world([_firm(1, "sort.records", 0, 1)], ["sort.records", need],
               {"founder_specs": [{"risk": 0.6, "competence": 0.1, "informed": 0.9, "patience": 0.9,
                                   "capital": 1_500_000}], "first_scan_day": 1.0, "scan_interval_days": 1.0},
               workers=5, civ={"worker_roles": ["developer"]}, n=2500)
    try:
        st = w.state
        for d in range(2, 22):
            w.advance_time(d * DAY)
            f = st.tables["founders"]["p:founder-00"]
            if f["venture"] and f["venture"]["stage"] == "operating":
                break
        f = st.tables["founders"]["p:founder-00"]
        v = f["venture"]
        assert v is not None and v["route"] == "hire" and v["stage"] == "operating", (f["phase"], v, f["estimates"])
        assert v["launched"] - v["t"] >= 2 * DAY                       # hiring and building took real time
        dev = v["dev"]
        skill = st.tables["npc_workers"][dev]["skills"]["developer"]
        assert v["tier"] == min(1, int(skill * 2))                     # what that developer can write
        assert st.tables["npc_workers"][dev]["employer"] == v["org"] and st.has("members", f"{v['org']}|{dev}")
        head = st.tables["commits"][st.tables["repos"][v["repo"]]["head"]]
        assert head["author"] == dev                                    # the developer committed the service
        c = st.tables["contracts"][v["dev_contract"]]
        assert c["status"] == "active" and any(cl["type"] == "payment" and cl["to"] == dev for cl in c["clauses"])
        w.advance_time(w.now + 2 * DAY)
        assert _served(w, need, w.now // DAY - 1) > 0
        _sound(w)
        _books_close(w)
    finally:
        w.close()


def test_a_company_with_sales_gets_a_loan_from_the_existing_lender_and_one_without_does_not():
    need = "text.extract"
    w = _world([_firm(1, "sort.records", 0, 1)], ["sort.records", need],
               {"founder_specs": [dict(GOOD, capital=700_000)], "first_scan_day": 1.0, "scan_interval_days": 1.0})
    try:
        st, mod = w.state, w.modules["entry"]
        w.advance_time(7 * DAY)
        fid = "p:founder-00"
        v = st.tables["founders"][fid]["venture"]
        assert v is not None and v["stage"] == "operating"
        lender = sorted(st.tables["npc_lenders"])[0]
        before = st.tables["accounts"][v["account"]]["balance"]

        def ask(amount):
            w._run_system("test-loan", lambda ctx: mod._seek_loan(ctx, fid, mod._rec(st, fid), mod._rec(st, fid)["venture"],
                                                                  amount, "bridging loan", ctx.rng("t")))
        ask(120_000)                 # well inside three times its verified 30-day sales
        ask(60_000_000)              # far beyond them (and beyond the lender's maximum)
        loans = st.tables["founders"][fid]["venture"]["loans"]
        assert [L["status"] for L in loans] == ["asked", "asked"]
        for _ in range(5):          # the lender answers within a day; the founder notices at its next look
            w.advance_time(w.now + DAY)
            loans = st.tables["founders"][fid]["venture"]["loans"]
            if all(L["status"] != "asked" for L in loans):
                break
        assert [L["status"] for L in loans] == ["granted", "refused"]
        c = st.tables["contracts"][loans[0]["contract"]]
        assert c["status"] == "active" and set(c["parties"]) == {v["org"], lender}
        paid = [t for t in st.tables["txns"].values() if t["ref"] == loans[0]["contract"] and t["kind"] == "contract"]
        assert len(paid) == 1 and sorted(paid[0]["postings"], key=lambda p: p[1])[1] == [v["account"], 120_000]
        assert st.has("sched", f"pay|{loans[0]['contract']}|1")           # the repayments are scheduled
        assert st.tables["contracts"][loans[1]["contract"]]["status"] == "rejected"
        assert st.tables["accounts"][v["account"]]["balance"] > before
        stats = st.tables["founders"][fid]["stats"]
        assert stats["loans_asked"] == 2 and stats["loans_granted"] == 1
        _sound(w)
    finally:
        w.close()


# ---------------------------------------------------------------------- the estimator itself
def test_the_estimate_orders_markets_sensibly_and_noise_shrinks_with_information():
    w = _world([_firm(1, "sort.records", 0, 1)], ["sort.records"], {"founder_specs": [dict(GOOD, capital=500_000)]})
    try:
        mod = w.modules["entry"]
        f = mod._rec(w.state, "p:founder-00")

        def est(n_sellers, price, u, sat, informed=1.0, seed=0, survey=True):
            g = dict(f, traits=dict(f["traits"], informed=informed))
            sellers = [{"host": f"h{i}", "seller": f"o:s{i}", "price": price, "rating": 3.0, "ratings": 5}
                       for i in range(n_sellers)]
            sig = {"need": "sort.records", "n": n_sellers, "sellers": sellers, "p_min": price if n_sellers else None,
                   "qgap": 0.3, "complaints": 0.0, "score": 1.0}
            sv = {"consumers_est": 2000, "requests_per_day_est": 20.0, "median_willingness_to_pay_cents": 5.0,
                  "unserved_share_est": u,
                  "offerings": {s["host"]: {"share_est": (1 - u) / n_sellers, "satisfaction_est": sat} for s in sellers}}
            return mod._estimate(g, sig, sv if survey else None, 150, np.random.default_rng(seed))
        empty = est(0, 0, 1.0, 0.5)
        gouged = est(1, 11, 0.5, 0.2)
        happy = est(3, 2, 0.02, 0.72)
        assert empty["monthly"] > gouged["monthly"] > 0 > happy["monthly"]
        assert empty["noise"] == 1.0                              # a fully informed founder has no estimate noise
        assert gouged["price"] <= 5 and happy["price"] == 2       # never above what the median customer pays
        spread = {i: np.std([np.log(est(1, 4, 0.5, 0.4, informed=i, seed=s)["noise"]) for s in range(200)])
                  for i in (0.9, 0.5, 0.1)}
        assert spread[0.9] < spread[0.5] < spread[0.1]
        no_survey = np.std([np.log(est(1, 4, 0.5, 0.4, informed=0.1, seed=s, survey=False)["noise"]) for s in range(200)])
        assert no_survey > spread[0.1]
    finally:
        w.close()


# ---------------------------------------------------------------------- admission harness
def test_the_entry_module_passes_the_admission_harness():
    from crucible.worldgen.admission import admit
    base = ["identity", "bank", "admin", "code", "search", "forum", "docs", "contracts", "business", "jobs", "cloud",
            "market"]
    rep = admit(["entry"], base=base, steps=300, seed=3,
                configs={"entry": {"founders": 3, "first_scan_day": 0.0, "scan_interval_days": 0.2}})
    assert rep.passed, (rep.checks, rep.errors[:5])
    assert set(rep.checks) >= {"schemas", "no_crashes", "invariants", "replay", "regeneration", "snapshot_fork",
                               "visibility", "protected_hooks"}


# ---------------------------------------------------------------------- code obtained from others
def _new_market_with_code(kind):
    """`text.extract` has customers and no seller; somebody else's implementation is available: as public
    open-source code (`open`) or as a paid licence on the market (`licence`)."""
    from crucible.worlds.digital_civilization import service_repo_files
    need = "text.extract"
    mid = {"risk": 0.5, "competence": 0.45, "informed": 0.9, "patience": 0.5, "capital": 900_000}
    w = _world([_firm(1, "sort.records", 0, 1)], ["sort.records", need],
               {"founder_specs": [mid], "first_scan_day": 1.0, "scan_interval_days": 1.0}, n=1500)
    w.system_call("admin.join", {"handle": "ada", "endowment": 100_000})

    def do(tool, **args):
        r = w.execute("p:ada", {"tool": tool, "args": args})
        assert r.ok, (tool, r.error)
        return r.result
    repo = do("code.create_repo", name="extract-lib", description=f"{need} service, reference quality",
              **({"visibility": "public", "license": "open"} if kind == "open" else {}))["repo"]
    do("code.commit", repo=repo, files=service_repo_files(need, 1, 3), message="best tier")
    listing = None
    if kind == "licence":
        listing = do("market.list", kind="code", title=f"{need} solver licence", price_cents=20_000, repo=repo,
                     need=need, description="Licence to use and fork our solver.")["listing"]
    return w, need, repo, listing


@pytest.mark.parametrize("kind", ["open", "licence"])
def test_a_founder_of_middling_competence_uses_public_or_licensed_code_instead_of_its_own(kind):
    w, need, src, listing = _new_market_with_code(kind)
    try:
        st = w.state
        ada = w.modules["bank"].primary_account(st, "p:ada")
        before = st.tables["accounts"][ada]["balance"]
        w.advance_time(4 * DAY)
        f = st.tables["founders"]["p:founder-00"]
        v = f["venture"]
        assert v is not None and v["stage"] == "operating", (f["phase"], f["estimates"], f["ventures"])
        assert v["route"] == ("fork" if kind == "open" else "licence") and v["tier"] is None
        code = w.modules["code"]
        assert code.files(st, v["repo"])["service.py"] == code.files(st, src)["service.py"]     # the better tier
        assert st.tables["repos"][v["repo"]]["owner"] == v["org"] and st.tables["repos"][v["repo"]]["visibility"] == "private"
        paid = st.tables["accounts"][ada]["balance"] - before
        if kind == "licence":
            assert paid == 20_000                                   # the licence was really bought, by the company
            order = next(o for o in st.tables["orders"].values() if o["listing"] == listing)
            assert order["buyer"] == v["org"] and order["status"] == "completed"
            assert any(g["grantee"] == v["org"] and g["cap"] == "code.read" and g["scope"] == src
                       for g in st.tables["grants"].values())
        else:
            assert paid == 0
        assert _served(w, need, 3) > 0
        _sound(w)
        _books_close(w)
    finally:
        w.close()
