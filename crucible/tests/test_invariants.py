"""The invariant monitor: clean on healthy worlds of several families, and every class of violation is detected
when state is deliberately corrupted in a copy."""
import pytest

import crucible

crucible.load_all_modules()
from crucible.actors import entry as _entry  # noqa: E402,F401
from crucible.core.kernel import World  # noqa: E402
from crucible.evaluation.invariants import InvariantMonitor, Violation, check_world  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400


# ---------------------------------------------------------------------- healthy worlds
@pytest.mark.parametrize("config,days", [
    ({"family": "tiny", "overrides": {"n_consumers": 300}}, 5),
    ({"family": "bazaar", "overrides": {"n_consumers": 800}}, 4),
    ({"family": "open_digital", "overrides": {"n_consumers": 400, "module_config": {"entry": {"first_scan_day": 2.0}}},
      "extra_modules": ["entry"]}, 5),
    ({"family": "open_industry", "overrides": {"n_shoppers": 3000}}, 6),
    ({"family": "engineering_lab"}, 2),
], ids=["tiny", "bazaar", "open_digital+entry", "open_industry", "engineering_lab"])
def test_healthy_worlds_stay_clean_under_continuous_monitoring(config, days):
    w = World.create(generate(config, 11))
    try:
        mon = InvariantMonitor(w).attach(every_s=6 * 3600)
        w.advance_time(days * DAY)
        assert mon.checks >= 4 * days - 1                       # it really ran throughout
        assert [str(v) for v in mon.violations] == []
        assert mon.check() == []
        s = mon.summary()
        assert s["violations"] == 0 and s["process_failures"] == {}
        if "accounts" in w.state.tables:                        # the monitor's own account of the money supply
            private = sum(a["balance"] for a in w.state.tables["accounts"].values() if a["kind"] != "system")
            assert s["money"]["private"] == private == s["money"]["recorded_net_inflow"]
            if "consumers" in w.modules:
                assert s["money"]["net_inflow_by_kind"].get("endowment", 0) > 0
                assert s["money"]["net_inflow_by_kind"].get("income", 0) > 0
        mon.detach()
        assert mon._on_event not in w.listeners
        print("MEASURED monitor:", config["family"], {"days": days, "checks": s["checks"], "wall_s": s["wall_s"],
                                                      "transactions": (s["money"] or {}).get("transactions")})
    finally:
        w.close()


# ---------------------------------------------------------------------- a world with one of everything
@pytest.fixture(scope="module")
def base():
    w = World.create(generate({"family": "tiny", "overrides": {"n_consumers": 300, "workers": 3}}, 4))
    st = w.state
    bank = w.modules["bank"]
    for h in ("ann", "bob"):
        w.system_call("admin.join", {"handle": h, "endowment": 500_000})
    a, b = bank.primary_account(st, "p:ann"), bank.primary_account(st, "p:bob")

    def do(actor, tool, **args):
        r = w.execute(actor, {"tool": tool, "args": args})
        assert r.ok, (tool, r.error)
        return r.result
    paid = do("p:ann", "bank.invoice", payer="bob", to_account=a, amount=1234)["invoice"]
    do("p:bob", "bank.pay_invoice", invoice=paid, from_account=b)
    open_inv = do("p:ann", "bank.invoice", payer="bob", to_account=a, amount=777)["invoice"]
    ctr = do("p:ann", "contracts.propose", parties=["p:ann", "p:bob"], title="retainer",
             clauses=[{"type": "payment", "from": "p:ann", "to": "p:bob", "amount": 700, "every_s": 7200, "count": 60,
                       "start_in_s": 0}])["contract"]
    do("p:bob", "contracts.sign", contract=ctr)
    mandate = do("p:ann", "bank.authorize_debit", from_account=a, payee="bob", max_per_period=5000)["mandate"]
    w._run_system("mint", lambda ctx: w.modules["market"].mint_goods(ctx, "p:ann", "widget", 10))
    lst = do("p:ann", "market.list", kind="goods", title="Widgets", price_cents=300, sku="widget")["listing"]
    order = do("p:bob", "market.buy", listing=lst, account=b, qty=2)["order"]
    mon = InvariantMonitor(w).attach(every_s=12 * 3600)
    w.advance_time(2 * DAY)
    assert mon.violations == [] and mon.check() == []
    firm = sorted(st.tables["npc_firms"])[0]
    info = {"a": a, "b": b, "paid_invoice": paid, "open_invoice": open_inv, "contract": ctr, "mandate": mandate,
            "order": order, "firm": firm, "host": st.tables["npc_firms"][firm]["host"]}
    yield w, info
    w.close()


def _copy(base, name, attach=False):
    w, info = base
    c = w.fork(f"corrupt-{name}")
    mon = InvariantMonitor(c)
    if attach:
        mon.attach(every_s=10 * DAY)
    return c, c.state, info, mon


def _kinds(vs):
    return {v.kind for v in vs}


def _one(vs, kind, where=None) -> Violation:
    hits = [v for v in vs if v.kind == kind and (where is None or v.where == where)]
    assert hits, (kind, where, sorted(_kinds(vs)))
    v = hits[0]
    assert isinstance(v.seq, int) and v.seq > 0 and v.t >= 0 and isinstance(v.amounts, dict) and v.where
    assert set(v.to_dict()) == {"kind", "where", "seq", "t", "amounts", "detail"}
    return v


def test_the_base_world_is_clean_and_a_copy_does_not_touch_it(base):
    w, info = base
    c, st, _, mon = _copy(base, "noop")
    st.patch("accounts", info["a"], balance=st.tables["accounts"][info["a"]]["balance"] + 1)
    assert mon.check() and check_world(w) == []
    c.close()


# ---------------------------------------------------------------------- ledger
def test_an_edited_balance_is_detected_with_the_amounts(base):
    c, st, info, mon = _copy(base, "balance")
    bal = st.tables["accounts"][info["a"]]["balance"]
    st.patch("accounts", info["a"], balance=bal + 500)
    vs = mon.check()
    v = _one(vs, "ledger.balance_mismatch", info["a"])
    assert v.amounts == {"balance": bal + 500, "postings": bal}
    assert _one(vs, "ledger.not_conserved").amounts == {"total": 500}
    m = _one(vs, "ledger.external_mismatch").amounts
    assert m["private"] - m["recorded_net_inflow"] == 500
    c.close()


def test_deleting_one_side_of_a_posting_is_detected(base):
    c, st, info, mon = _copy(base, "posting")
    rid = sorted(r for r in st.tables["acct_txns"] if r.endswith("|" + info["b"]))[0]
    amount = st.tables["acct_txns"][rid]["amount"]
    st.delete("acct_txns", rid)
    vs = mon.check()
    assert _one(vs, "ledger.posting_missing", rid).amounts == {"amount": amount}
    assert _one(vs, "ledger.balance_mismatch", info["b"])
    c.close()


def test_an_unbalanced_transaction_and_a_changed_or_orphan_row_are_detected(base):
    c, st, info, mon = _copy(base, "unbalanced")
    tid = sorted(st.tables["txns"])[-1]
    tx = dict(st.tables["txns"][tid])
    acct, d = tx["postings"][0]
    st.put("txns", tid, dict(tx, postings=[[acct, d - 10]] + [list(p) for p in tx["postings"][1:]]))
    st.put("acct_txns", "tx_999999|" + info["a"], {"acct": info["a"], "txn": "tx_999999", "amount": 5, "t": 0})
    vs = mon.check()
    assert _one(vs, "ledger.unbalanced_txn", tid).amounts["sum"] == -10
    assert _one(vs, "ledger.posting_mismatch", f"{tid}|{acct}").amounts == {"posting": d - 10, "row": d}
    assert _one(vs, "ledger.orphan_posting", "tx_999999|" + info["a"]).amounts == {"amount": 5}
    c.close()


def test_an_overdrawn_account_is_detected_even_when_the_ledger_is_otherwise_consistent(base):
    c, st, info, mon = _copy(base, "floor")
    a, b = info["a"], info["b"]
    x = st.tables["accounts"][b]["balance"] + 1000            # a balanced, fully recorded transfer that overdraws b
    st.put("txns", "tx_888888", {"t": c.now, "postings": [[a, x], [b, -x]], "kind": "transfer", "memo": "", "ref": None,
                                 "by": "p:bob"})
    for acct, d in ((a, x), (b, -x)):
        st.put("acct_txns", f"tx_888888|{acct}", {"acct": acct, "txn": "tx_888888", "amount": d, "t": c.now})
        st.patch("accounts", acct, balance=st.tables["accounts"][acct]["balance"] + d)
    vs = mon.check()
    assert _one(vs, "ledger.below_floor", b).amounts == {"balance": -1000, "floor": 0}
    assert not [k for k in _kinds(vs) if k.startswith("ledger.") and k != "ledger.below_floor"]
    c.close()


def test_money_created_from_a_system_account_without_a_record_is_detected(base):
    c, st, info, mon = _copy(base, "external")
    st.patch("accounts", "acct_genesis", balance=st.tables["accounts"]["acct_genesis"]["balance"] - 9000)
    st.patch("accounts", info["a"], balance=st.tables["accounts"][info["a"]]["balance"] + 9000)
    vs = mon.check()
    assert "ledger.not_conserved" not in _kinds(vs)           # the total still sums to zero ...
    m = _one(vs, "ledger.external_mismatch").amounts          # ... but nothing recorded says where it came from
    assert m["private"] - m["recorded_net_inflow"] == 9000
    c.close()


# ---------------------------------------------------------------------- contracts, claims, invoices, escrow
def test_contract_states_and_payments_are_checked_against_the_ledger(base):
    c, st, info, mon = _copy(base, "contract")
    cid = info["contract"]
    k = st.tables["contracts"][cid]
    n = k["paid"]["0"]
    assert k["status"] == "active" and n >= 20
    st.patch("contracts", cid, paid={"0": n + 1})
    v = _one(mon.check(), "contract.payments_mismatch", f"{cid}#0")
    assert v.amounts["recorded"] == n + 1 and v.amounts["ledger"] == n and v.amounts["ledger_total"] == 700 * n
    st.patch("contracts", cid, paid={"0": n}, signed=["p:ann"])
    assert _one(mon.check(), "contract.unsigned_active", cid).amounts == {"missing": ["p:bob"]}
    st.patch("contracts", cid, signed=["p:ann", "p:bob"])
    assert mon.check() == []
    st.delete("sched", f"pay|{cid}|0")
    v = _one(mon.check(), "contract.payment_unscheduled", f"{cid}#0")
    assert v.amounts == {"recorded": n, "due": 60, "amount": 700}
    st.patch("contracts", cid, status="frozen")
    assert _one(mon.check(), "contract.illegal_state", cid)
    st.patch("contracts", cid, status="completed")
    assert _one(mon.check(), "contract.completed_unpaid", f"{cid}#0").amounts == {"recorded": n, "due": 60}
    c.close()


def test_a_ledger_payment_that_the_contract_does_not_record_is_detected(base):
    c, st, info, mon = _copy(base, "contract-ledger")
    cid = info["contract"]
    tid = sorted(t for t, x in st.tables["txns"].items() if x["ref"] == cid)[-1]
    st.patch("txns", tid, ref="ctr_777777")
    vs = mon.check()
    assert _one(vs, "contract.payments_mismatch", f"{cid}#0").amounts["ledger"] == st.tables["contracts"][cid]["paid"]["0"] - 1
    assert _one(vs, "contract.payment_without_contract", "ctr_777777#0").amounts == {"ledger": 1, "ledger_total": 700}
    c.close()


def test_claims_invoices_mandates_and_escrow_must_agree_with_what_they_stand_for(base):
    c, st, info, mon = _copy(base, "instruments")
    st.patch("invoices", info["open_invoice"], status="paid")
    assert _one(mon.check(), "invoice.paid_without_payment", info["open_invoice"]).amounts == {"amount": 777}
    st.patch("invoices", info["open_invoice"], status="open")
    st.put("claims", "claim_900001", {"debtor": info["firm"], "creditor": "p:ann", "amount": 5000, "source": "test",
                                      "status": "open", "t": c.now - 20 * DAY, "paid": 0})
    v = _one(mon.check(), "claim.unresolved_past_grace", "claim_900001")
    assert v.amounts["amount"] == 5000 and v.amounts["age_s"] == 20 * DAY and v.amounts["grace_s"] == 14 * DAY
    st.patch("claims", "claim_900001", status="settled")
    assert _one(mon.check(), "claim.inconsistent", "claim_900001").amounts == {"amount": 5000, "paid": 0}
    st.delete("claims", "claim_900001")
    st.patch("mandates", info["mandate"], used=6000)
    assert _one(mon.check(), "mandate.over_limit", info["mandate"]).amounts == {"used": 6000, "max": 5000}
    st.patch("mandates", info["mandate"], used=0)
    st.patch("orders", info["order"], amount=st.tables["orders"][info["order"]]["amount"] + 50)
    v = _one(mon.check(), "escrow.mismatch", "acct_escrow")
    assert v.amounts["open_orders"] - v.amounts["balance"] == 50
    st.patch("orders", info["order"], amount=st.tables["orders"][info["order"]]["amount"] - 50)
    bid = sorted(st.tables["bounties"])[0]
    st.patch("bounties", bid, reward=st.tables["bounties"][bid]["reward"] - 7)
    v = _one(mon.check(), "escrow.mismatch", "acct_bounties")
    assert v.amounts["balance"] - v.amounts["open_rewards"] == 7
    c.close()


# ---------------------------------------------------------------------- compute
def test_capacity_overdraw_and_unpaid_compute_are_detected(base):
    c, st, info, mon = _copy(base, "compute")
    dep = st.tables["deployments"][st.tables["hosts"][info["host"]]["deployment"]]
    iid = dep["instance"]
    inst = st.tables["instances"][iid]
    prov = st.tables["providers"][inst["provider"]]
    used = sum(i["cores"] for i in st.tables["instances"].values() if i["provider"] == inst["provider"]
               and i["status"] == "running")
    st.patch("providers", inst["provider"], cores=used - 1)
    assert _one(mon.check(), "compute.oversubscribed", inst["provider"]).amounts == {"used": used, "capacity": used - 1}
    st.patch("providers", inst["provider"], cores=prov["cores"])
    st.patch("instances", iid, used_cu=10 * inst["cores"] * 100_000.0, hour=c.now // 3600)
    v = _one(mon.check(), "compute.instance_overused", iid)
    assert v.amounts == {"used_cu": 10 * inst["cores"] * 100_000.0, "capacity_cu": inst["cores"] * 100_000}
    st.patch("instances", iid, used_cu=0.0)
    st.patch("instances", iid, status="suspended", suspended_at=c.now - 5 * DAY)
    v = _one(mon.check(), "compute.unpaid_beyond_grace", iid)
    assert v.amounts == {"suspended_for_s": 5 * DAY, "grace_s": 72 * 3600}
    st.patch("instances", iid, status="terminated")
    assert _one(mon.check(), "compute.live_on_dead_instance", st.tables["hosts"][info["host"]]["deployment"])
    st.patch("instances", iid, status="running", suspended_at=None)
    assert mon.check() == []
    # a service that keeps running although its bills stopped being paid
    for tid, tx in list(st.tables["txns"].items()):
        if tx["kind"] == "compute" and tx["memo"] == f"instance {iid}" and tx["t"] > c.now - 12 * 3600:
            st.patch("txns", tid, memo="instance someone-else")
    v = _one(mon.check(), "compute.unbilled_instance", iid)
    assert v.amounts["unpaid_for_s"] >= 11 * 3600 and v.amounts["hourly_cents"] == inst["price"] * inst["cores"]
    c.close()


def test_a_bankrupt_company_that_still_operates_is_detected(base):
    c, st, info, mon = _copy(base, "defunct")
    st.patch("companies", info["firm"], status="bankrupt")
    wheres = {v.where for v in mon.check() if v.kind == "company.defunct_operating"}
    assert {info["firm"], f"{info['firm']}:listings", f"{info['firm']}:accounts"} <= wheres
    st.patch("orgs", info["firm"], status="bankrupt")
    wheres = {v.where for v in mon.check() if v.kind == "company.defunct_operating"}
    assert f"{info['firm']}:{info['host']}" in wheres and any(w_.startswith(f"{info['firm']}:inst_") for w_ in wheres)
    st.patch("companies", info["firm"], status="registered")
    st.patch("orgs", info["firm"], status="active")
    row = sorted(k for k in st.tables["shares"] if k.startswith(info["firm"] + "|"))[0]
    st.patch("shares", row, n=st.tables["shares"][row]["n"] - 1)
    assert _one(mon.check(), "equity.shares_mismatch", info["firm"]).amounts == {"shares": 999_999, "issued": 1_000_000}
    c.close()


# ---------------------------------------------------------------------- scheduler, goods, kernel head
def test_overdue_unqueued_and_vanished_jobs_are_detected(base):
    c, st, info, mon = _copy(base, "sched", attach=True)
    job = dict(st.tables["sched"]["market.sweep"])
    st.patch("sched", "market.sweep", at=c.now - 100)
    vs = mon.check()
    assert _one(vs, "sched.overdue", "market.sweep").amounts == {"at": c.now - 100, "now": c.now, "late_s": 100}
    assert _one(vs, "sched.unqueued", "market.sweep")           # the run queue still holds the old time
    st.put("sched", "market.sweep", job)
    assert mon.check() == []
    st.delete("sched", "cloud.hourly")
    assert _one(mon.check(), "sched.periodic_job_missing", "cloud.hourly")
    c.close()


def test_goods_and_the_log_head_are_checked(base):
    c, st, info, mon = _copy(base, "goods")
    key = "p:ann|widget"
    have = st.tables["inventory"][key]["qty"]
    st.patch("inventory", key, qty=have + 3)
    v = _one(mon.check(), "inventory.goods_not_conserved", "widget")
    assert v.amounts == {"existing": 13, "minted": 10}
    st.patch("inventory", key, qty=-1)
    assert _one(mon.check(), "inventory.negative", key).amounts == {"qty": -1}
    st.patch("inventory", key, qty=have)
    st.meta["seq"] = st.meta["seq"] + 1
    assert _one(mon.check(), "kernel.head_mismatch").amounts["log_seq"] == c.seq
    c.close()


def test_the_modules_own_checks_are_run_too(base):
    c, st, info, mon = _copy(base, "module")
    person = sorted(st.tables["npc_workers"])[0]
    st.patch("npc_workers", person, employer="o:nowhere")      # only the npcs module knows this rule
    v = _one(mon.check(), "module.npcs")
    assert person in v.detail
    assert InvariantMonitor(c, run_module_checks=False).check() == []
    c.close()


# ---------------------------------------------------------------------- what only a following monitor can see
def test_a_violation_carries_the_sequence_number_of_the_event_that_posted_the_transaction(base):
    c, st, info, mon = _copy(base, "seq", attach=True)
    r = c.execute("p:ann", {"tool": "bank.transfer", "args": {"from_account": info["a"], "to": "bob", "amount": 4321}})
    assert r.ok
    tid = r.result["txn"]
    ev = next(e for e in c.log if e.type == "bank.txn" and e.payload["id"] == tid)
    st.patch("txns", tid, postings=[[info["a"], -4321], [info["b"], 4000]])
    v = _one(mon.check(), "ledger.unbalanced_txn", tid)
    assert v.seq == ev.seq and v.seq < c.seq and v.amounts["sum"] == -321
    c.close()


def test_a_process_that_keeps_failing_is_reported_whatever_else_looks_healthy(base):
    c, st, info, mon = _copy(base, "proc", attach=True)

    def boom(ctx):
        raise RuntimeError("hidden crash")
    c.processes["test.boom"] = boom
    c._run_system("schedule", lambda ctx: ctx.schedule(ctx.now + 3600, "test.boom", {}, period=3600, key="test.boom"))
    c.advance_time(c.now + 2 * 3600)
    assert mon.check() == []                                    # two failures: not yet "repeatedly"
    c.advance_time(c.now + 3 * 3600)
    first = next(e for e in c.log if e.type == "proc.failed")
    v = _one(mon.check(), "process.repeated_failure", "test.boom")
    assert v.amounts["failures"] == 5 and v.seq == first.seq and "hidden crash" in v.detail
    day = c.now // DAY - 1                                      # demand is still being served all the while
    assert sum(r["ok"] for k, r in st.scan("svc_days") if k.endswith(f"|{day}")) > 0
    assert mon.summary()["process_failures"]["test.boom"]["count"] == 5
    c.close()


def test_industrial_stock_energy_and_reserves_are_checked():
    w = World.create(generate({"family": "open_industry", "overrides": {"n_shoppers": 3000}}, 2))
    try:
        mon = InvariantMonitor(w).attach(every_s=DAY)
        w.advance_time(5 * DAY)
        st = w.state
        assert mon.violations == [] and len(mon._stock_delta) > 0      # stock really moved, all of it explained
        key = sorted(k for k, s in st.tables["stock"].items() if s["qty"] > 0)[0]
        qty = st.tables["stock"][key]["qty"]
        st.patch("stock", key, qty=qty + 5)                            # stock appears from nowhere
        v = [x for x in mon.check() if x.kind == "inventory.unexplained_change"]
        assert len(v) == 1 and v[0].where == key
        assert v[0].amounts["actual"] - v[0].amounts["expected"] == 5 and v[0].seq <= w.seq
        st.patch("stock", key, qty=-2)
        assert any(x.kind == "inventory.negative" and x.where == key for x in mon.check())
        st.patch("stock", key, qty=qty)
        site = sorted(st.tables["sites"])[0]
        reserve = st.tables["sites"][site]["reserve"]
        st.patch("sites", site, reserve=-1)
        assert any(x.kind == "inventory.negative" and x.where == f"site {site}" for x in mon.check())
        st.patch("sites", site, reserve=reserve)
        gk = sorted(st.tables["grid_days"])[-1]
        g = st.tables["grid_days"][gk]
        cap = st.tables["regions"][g["region"]]["grid"]["capacity"]
        st.patch("grid_days", gk, used=cap + 10.0)
        v = [x for x in mon.check() if x.kind == "energy.over_capacity"]
        assert v and v[0].amounts == {"used": cap + 10.0, "capacity": cap}
        st.patch("grid_days", gk, used=g["used"])
        assert mon.check() == []
    finally:
        w.close()
