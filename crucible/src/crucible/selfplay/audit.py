"""Collusion and reward-inflation audit for self-play.

In a kernel-hosted world the participants of a match are ordinary principals: they can pay each other, found
organisations and move money between them. Self-play results must not be movable by any of that, and a pair of
learners that does well together must be told apart from a pair that merely takes more from the customers. Three
things are established here (and tested in `tests/test_selfplay.py`):

1. **Transfers cannot move the objective** (`transfer_invariance`). The self-play objective of a seat holder is the
   facility metric `rig_net`: what the facility's independent demand paid the seat, minus the seat's operating
   cost. It is computed from the facility's own reports, never from a bank balance. The same match is played twice
   in kernel worlds, once with a schedule of payments between the holders, their organisations and third parties
   (one direction, the other, rings), once without: `rig_net` of every holder is identical to the cent, while the
   bank balances differ by exactly the payments. (If a holder is so short of funds that it cannot pay for its
   settings, money from a partner changes what it can *run*; that is a real transfer of resources, the extra
   revenue still comes from the independent demand, and `binding` in the result says when this happened.)

2. **Conservation** (`conservation`, `ledger_conservation`). Everything the seats earned was paid by the
   facility's demand account, which is funded only from outside the world at a fixed rate: the total revenue of
   all seats never exceeds the demand income, in every match and at every tick of a kernel world.

3. **What a pair's joint behaviour is** (`classify_pair`). A pair is compared with a matched reference: the same
   seats in the same worlds with the same noise, playing without each other's cooperation. From the engine's
   private diagnostics (customers actually served and what they paid; work actually completed) the pair is

   * `value_creating_cooperation`: the joint net is above the reference, no seat is worse off, and more value was
     actually delivered (more customers served, more work completed in time);
   * `tacit_collusion`: both nets are above the reference although no more value was delivered, and customers paid
     more per unit: the gain is extracted from the demand side (for instance both firms of a duopoly raising price);
   * `mutual_cost_restraint`: both nets are above the reference, no more value delivered, customers not charged
     more per unit: the gain is rivalry cost the seats stopped burning (for instance an advertising truce);
   * `one_sided`: the joint net is above the reference but a seat is worse off;
   * `no_joint_gain`: the joint net is not above the reference.

   Circular manipulation (raising a result by passing money around) is reported as impossible with the evidence:
   the objective has no term a payment between participants could enter. Nothing is forbidden: collusive
   strategies are recorded with their evidence, so they can be studied.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from ..research import stats
from ..research.learner import EnvSpec
from ..research.tasks import Task
from .arena import KernelSelfPlayEnv, summarise

CLASSES = ("value_creating_cooperation", "tacit_collusion", "mutual_cost_restraint", "one_sided", "no_joint_gain")


# ====================================================================== 1. transfers
def open_org(env: KernelSelfPlayEnv, seat: int, handle: str, fund: int = 0) -> dict:
    """Seat `seat`'s holder founds an organisation, opens a bank account for it and (optionally) moves `fund`
    cents of its own money into it. Returns {"org", "account"}."""
    org = env._call(seat, "id.create_org", {"handle": handle, "name": handle})["org"]
    acct = env._call(seat, "bank.open_account", {"name": "operating", "owner": org})["account"]
    if fund > 0:
        env.transfer(seat, acct, fund, "capitalisation")
    return {"org": org, "account": acct}


def _run_kernel(task: Task, schedule: Callable[[int, int], np.ndarray | None], ticks: int, spec: EnvSpec | None,
                funds: Sequence[int] | None, payments: Sequence[dict] | None, extra: Sequence[str]) -> dict:
    env = KernelSelfPlayEnv(task, spec, funds=funds, extra_handles=extra)
    env.reset()
    S = env.n_seats
    orgs: dict[str, dict] = {}
    extra_acct = {}
    for h, pid in env.extra.items():
        accts = env.world.modules["bank"].primary_account(env.world.state, pid)
        extra_acct[h] = accts
    moved = []
    by_tick: dict[int, list[dict]] = {}
    for p in payments or []:
        by_tick.setdefault(int(p["t"]), []).append(p)
    status = [[] for _ in range(S)]
    for t in range(ticks):
        for p in by_tick.get(t, []):
            src, dst, amt = p["from"], p["to"], int(p["amount"])

            def account_of(x):
                if isinstance(x, int):
                    return env.account(x)
                if x in orgs:
                    return orgs[x]["account"]
                return extra_acct[x]
            if isinstance(src, str) and src.startswith("org:") and src not in orgs:
                orgs[src] = dict(open_org(env, int(p["owner"]), src[4:]), owner=int(p["owner"]))
            if isinstance(dst, str) and dst.startswith("org:") and dst not in orgs:
                orgs[dst] = dict(open_org(env, int(p["dst_owner"]), dst[4:]), owner=int(p["dst_owner"]))
            if isinstance(src, int):
                actor_seat, from_acct = src, env.account(src)
                env._call(actor_seat, "bank.transfer", {"from_account": from_acct, "to": account_of(dst),
                                                        "amount": amt, "memo": p.get("memo", "side payment")})
            elif src in orgs:                       # the organisation's owner pays from the organisation's account
                env._call(orgs[src]["owner"], "bank.transfer", {"from_account": orgs[src]["account"],
                                                                "to": account_of(dst), "amount": amt,
                                                                "memo": p.get("memo", "side payment")})
            else:                                   # a third participant pays
                res = env.world.execute(env.extra[src], {"tool": "bank.transfer", "args": {
                    "from_account": extra_acct[src], "to": account_of(dst), "amount": amt,
                    "memo": p.get("memo", "side payment")}})
                if not res.ok:
                    raise RuntimeError(f"transfer by {src} failed: {res.error}")
            moved.append({"t": t, "from": src, "to": dst, "amount": amt})
        _, _, _, info = env.step([schedule(s, t) for s in range(S)])
        for s in range(S):
            status[s].append(info["status"][s][0])
    w = env.world
    rig = w.state.tables["rigs"]["r1"]
    reports = [w.modules["rig"].reports(w, "r1", s) for s in range(S)]
    return {"rig_net": [env.rig_net(s) for s in range(S)],
            "net_from_reports": [float(sum(r["revenue_cents"] - r["cost_cents"] for r in reports[s])) for s in range(S)],
            "revenue": [float(sum(r["revenue_cents"] for r in reports[s])) for s in range(S)],
            "balance": [env.balance(s) for s in range(S)], "funds0": list(env.funds0),
            "status": status, "moved": moved, "funded": int(rig["funded"]), "paid": int(rig["paid"]),
            "charged": int(rig["charged"]), "invariants": w.check_invariants(), "digest": w.digest(),
            "orgs": {k: w.modules["bank"].balance(w.state, v["account"]) for k, v in orgs.items()},
            "env": env}


def transfer_invariance(task: Task, schedule: Callable[[int, int], np.ndarray | None], payments: Sequence[dict],
                        ticks: int = 30, spec: EnvSpec | None = None, funds: Sequence[int] | None = None,
                        extra: Sequence[str] = ()) -> dict:
    """Play the same match twice in kernel worlds, with and without `payments`, and compare the objective.

    `schedule(seat, t)` gives the action vector of a seat at tick t (None: keep). `payments` is a list of
    `{"t": tick, "from": X, "to": Y, "amount": cents}` executed with `bank.transfer` before that tick, where X
    and Y are a seat number, the handle of a third participant listed in `extra`, or `"org:<handle>"` for an
    organisation (founded on first use by seat `owner` / `dst_owner`).

    Returns both runs' `rig_net`, the per-seat difference (`objective_moved_cents`, all zeros when no holder's
    funds bind), the net payment received by each seat, the change of each seat's bank balance, and `binding`:
    whether any seat ran at fallback or idle in either run (the one way money can change what a seat runs).
    """
    base = _run_kernel(task, schedule, ticks, spec, funds, None, extra)
    paid = _run_kernel(task, schedule, ticks, spec, funds, payments, extra)
    S = len(base["rig_net"])
    recv = [0] * S
    for m in paid["moved"]:
        if isinstance(m["from"], int):
            recv[m["from"]] -= m["amount"]
        if isinstance(m["to"], int):
            recv[m["to"]] += m["amount"]
    binding = any(st != "ok" for run in (base, paid) for seat in run["status"] for st in seat)
    return {"rig_net_without": base["rig_net"], "rig_net_with": paid["rig_net"],
            "objective_moved_cents": [paid["rig_net"][s] - base["rig_net"][s] for s in range(S)],
            "payments": len(paid["moved"]), "payments_total_cents": int(sum(m["amount"] for m in paid["moved"])),
            "net_received_cents": recv,
            "balance_change_cents": [paid["balance"][s] - base["balance"][s] for s in range(S)],
            "revenue_without": base["revenue"], "revenue_with": paid["revenue"], "binding": binding,
            "paid_by_demand": [base["paid"], paid["paid"]], "funded": [base["funded"], paid["funded"]],
            "invariants": base["invariants"] + paid["invariants"], "orgs": paid["orgs"],
            "runs": (base, paid)}


# ====================================================================== 2. conservation
def conservation(records: Sequence[dict]) -> dict:
    """Arena match records: what the seats were paid against what the independent demand could pay. A match is a
    violation if the seats together received more than (a) the demand income the facility was given over the
    match, (b) the engine's own bound on what its demand side can have paid by then (`Engine.revenue_bound`:
    for a market, the customers' starting budgets plus their income), or (c) what the engine's private
    diagnostics say the customers actually paid the seats (for a technical system: the value of the work
    completed, up to the facility's rounding of every payment to whole cents). There must be none."""
    bad, worst_income, worst_bound, worst_round = [], 0.0, 0.0, float("-inf")
    for r in records:
        seats = float(sum(s["revenue_cents"] for s in r["seats"]))
        why = []
        worst_income = max(worst_income, seats / max(1.0, float(r["demand_income_cents"])))
        if seats > r["demand_income_cents"] + 1e-6:
            why.append("above the demand income")
        if abs(seats - float(r["paid_by_demand_cents"])) > 1e-6:
            why.append("seat revenue differs from what the demand account paid")
        rb = r.get("revenue_bound_cents")
        if rb is not None:
            worst_bound = max(worst_bound, seats / max(1.0, float(rb)))
            if seats > rb + 1e-6:
                why.append("above the engine's revenue bound")
        if r["engine"] == "abm":
            if seats > r["private"]["paid_seats"] + 1e-6:
                why.append("above what customers paid the seats")
            if r["private"]["paid_market"] > (rb if rb is not None else float("inf")) + 1e-6:
                why.append("all firms were paid more than the customers could spend")
        if r["engine"] == "techsys":
            # the engine's revenue is fractional; the facility pays whole cents, rounded to the nearest, per seat
            # and tick: the ledger can differ from the value of the completed work by at most half a cent each
            slack = 0.5 * len(r["seats"]) * r["ticks"]
            excess = seats - r["private"].get("value_completed", float("inf"))
            worst_round = max(worst_round, excess)
            if excess > slack + 1e-6:
                why.append("above the value of the work completed")
        if why:
            bad.append({"task": r["task"], "noise": r.get("noise"), "seats_revenue": seats, "why": why})
    return {"matches": len(records), "violations": bad, "max_revenue_over_demand_income": worst_income,
            "max_revenue_over_engine_bound": worst_bound if worst_bound else None,
            "max_cent_rounding_excess_cents": None if worst_round == float("-inf") else float(worst_round)}


def ledger_conservation(world, rig: str = "r1") -> dict:
    """A kernel world: what the facility's demand account received from outside, what it paid to seat holders,
    and what the holders' reports say they earned. `ok` requires paid <= funded, a demand balance of exactly
    funded - paid that was never negative, and the reports adding up to what was paid."""
    st = world.state
    rec = st.tables["rigs"][rig]
    bank = world.modules["bank"]
    stats_rows = st.find("rig_stats", "rig", rig)
    earned = int(sum(r["revenue"] for r in stats_rows))
    bal = bank.balance(st, rec["account"])
    return {"funded": int(rec["funded"]), "paid": int(rec["paid"]), "charged": int(rec["charged"]),
            "earned_by_holders": earned, "demand_balance": int(bal),
            "ok": bool(rec["paid"] <= rec["funded"] and bal == rec["funded"] - rec["paid"] and bal >= 0
                       and earned == rec["paid"])}


def bilateral_transfers(world, principals: Sequence[str]) -> dict:
    """Ledger transfers (`bank.transfer`) in which an account of one of `principals` (or of an organisation one
    of them owns) pays an account of another. Recorded, not forbidden."""
    st = world.state
    owner_of = {aid: a["owner"] for aid, a in st.tables["accounts"].items()}
    group = {}
    for p in principals:
        group[p] = p
        for m in st.find("members", "person", p) if "members" in st.tables else []:
            group[m["org"]] = p
    rows, total = [], 0
    for tid in st.find_ids("txns", "kind", "transfer"):
        tx = st.tables["txns"][tid]
        src = [a for a, amt in tx["postings"] if amt < 0]
        dst = [a for a, amt in tx["postings"] if amt > 0]
        if len(src) == 1 and len(dst) == 1:
            a, b = group.get(owner_of.get(src[0])), group.get(owner_of.get(dst[0]))
            if a is not None and b is not None and a != b:
                amt = int(sum(x for _, x in tx["postings"] if x > 0))
                rows.append({"t": tx["t"], "from": a, "to": b, "amount": amt})
                total += amt
    return {"transfers": rows, "total_cents": total}


# ====================================================================== 3. what a pair's behaviour is
def _value(private: dict, engine: str) -> tuple[float | None, float | None, str]:
    """(value delivered by the seats' side, price paid per unit, what the value is) from private quantities."""
    if engine == "abm":
        units = private.get("units_seats")
        price = private["paid_seats"] / units if units else None
        return units, price, "customers served by the seats (units bought)"
    if engine == "techsys":
        return private.get("completed"), None, "jobs completed while still wanted"
    return None, None, "unknown"


def matched_reference(seat0_with_noncoop: Sequence[dict], seat1_with_noncoop: Sequence[dict]) -> list[dict]:
    """The reference of `classify_pair` for a pair (A in seat 0, B in seat 1): what each of them gets with a
    matched partner that does not cooperate. `seat0_with_noncoop` are the records of (A, non-cooperating
    partner), `seat1_with_noncoop` those of (non-cooperating partner, B), on the same worlds and noise in the
    same order. The result has, per match, seat 0's numbers from the first and seat 1's from the second; the
    private quantities are the mean of the two matches (each is a whole-facility count)."""
    if len(seat0_with_noncoop) != len(seat1_with_noncoop):
        raise ValueError("the two reference lists must be matched match for match")
    out = []
    for r0, r1 in zip(seat0_with_noncoop, seat1_with_noncoop):
        if r0["task"] != r1["task"] or r0.get("noise") != r1.get("noise"):
            raise ValueError("reference matches must be on the same world and noise")
        seats = [r0["seats"][0], r1["seats"][1]]
        keys = sorted(set(r0["private"]) | set(r1["private"]))
        out.append({"task": r0["task"], "noise": r0.get("noise"), "engine": r0["engine"], "ticks": r0["ticks"],
                    "seats": seats, "joint_net_per_tick": seats[0]["net_per_tick"] + seats[1]["net_per_tick"],
                    "private": {k: 0.5 * (r0["private"].get(k, 0.0) + r1["private"].get(k, 0.0)) for k in keys}})
    return out


def classify_pair(pair: Sequence[dict], reference: Sequence[dict], tol: float = 0.02,
                  transfers_cents: float = 0.0) -> dict:
    """Classify a pair's joint behaviour from match records (`arena.play` output).

    `pair`: the two individuals playing together. `reference`: matched play without the cooperation in question
    (the same seats and worlds; ideally the same noise, in the same order, in which case the differences are
    paired). `tol`: a change smaller than this fraction of the reference magnitude counts as no change.
    `transfers_cents`: payments observed between the pair (kernel worlds; reported, never added to a net).

    Returns the class (see the module docstring), the evidence (joint and per-seat nets, value delivered, price
    per unit, market-wide units where the engine has rivals) for both sides with paired intervals where the
    records are matched, and the statement on circular manipulation.
    """
    if not pair or not reference:
        raise ValueError("classify_pair needs match records for the pair and for the reference")
    engine = pair[0]["engine"]
    sp, sr = summarise(pair), summarise(reference)
    S = len(sp["net_per_tick"])
    matched = len(pair) == len(reference) and all(a.get("noise") == b.get("noise") and a["task"] == b["task"]
                                                  for a, b in zip(pair, reference))
    vp, pp, what = _value(sp["private"], engine)
    vr, pr, _ = _value(sr["private"], engine)

    def rel(a, b):
        return None if a is None or b is None else float((a - b) / max(abs(b), 1e-9))

    d_net = [sp["net_per_tick"][s] - sr["net_per_tick"][s] for s in range(S)]
    d_joint = sp["joint_net_per_tick"] - sr["joint_net_per_tick"]
    scale = max(abs(sr["joint_net_per_tick"]), max(abs(x) for x in sr["net_per_tick"]), 1e-9)
    up = [d > tol * scale for d in d_net]
    down = [d < -tol * scale for d in d_net]
    value_up = vp is not None and vr is not None and (vp - vr) > tol * max(abs(vr), 1e-9)
    price_up = pp is not None and pr is not None and (pp - pr) > tol * max(abs(pr), 1e-9)
    if d_joint <= tol * scale:
        cls = "no_joint_gain"
    elif any(down):
        cls = "one_sided"
    elif value_up:
        cls = "value_creating_cooperation"
    elif all(up) and price_up:
        cls = "tacit_collusion"
    elif all(up):
        cls = "mutual_cost_restraint"
    else:                                            # joint gain, nobody clearly worse, not both clearly better
        cls = "value_creating_cooperation" if value_up else "one_sided"
    ev: dict[str, Any] = {
        "joint_net_per_tick": {"pair": sp["joint_net_per_tick"], "reference": sr["joint_net_per_tick"], "diff": d_joint},
        "net_per_tick": {"pair": sp["net_per_tick"], "reference": sr["net_per_tick"], "diff": d_net},
        "value_delivered": {"what": what, "pair": vp, "reference": vr, "relative_change": rel(vp, vr)},
        "price_per_unit": {"pair": pp, "reference": pr, "relative_change": rel(pp, pr)},
    }
    if engine == "abm":
        ev["market_units"] = {"pair": sp["private"].get("units_market"), "reference": sr["private"].get("units_market"),
                              "relative_change": rel(sp["private"].get("units_market"), sr["private"].get("units_market"))}
        ev["customer_spend_at_seats"] = {"pair": sp["private"].get("paid_seats"),
                                         "reference": sr["private"].get("paid_seats")}
        if "pool" in sp["private"]:
            ev["shared_pool_level"] = {"pair": sp["private"]["pool"], "reference": sr["private"]["pool"]}
    if engine == "techsys":
        ev["value_completed"] = {"pair": sp["private"].get("value_completed"),
                                 "reference": sr["private"].get("value_completed")}
        if sp["private"].get("pool_util") is not None:
            ev["shared_pool_utilisation"] = {"pair": sp["private"].get("pool_util"),
                                             "reference": sr["private"].get("pool_util")}
    if matched and len(pair) >= 2:
        ev["paired"] = {
            "joint_net_per_tick": stats.paired_diff([r["joint_net_per_tick"] for r in pair],
                                                    [r["joint_net_per_tick"] for r in reference]),
            "net_per_tick": [stats.paired_diff([r["seats"][s]["net_per_tick"] for r in pair],
                                               [r["seats"][s]["net_per_tick"] for r in reference]) for s in range(S)]}
        key = "units_seats" if engine == "abm" else ("completed" if engine == "techsys" else None)
        if key is not None:
            ev["paired"]["value_delivered"] = stats.paired_diff([r["private"][key] for r in pair],
                                                                [r["private"][key] for r in reference])
    return {"class": cls, "matched": matched, "n": [len(pair), len(reference)], "tolerance": tol, "evidence": ev,
            "flags": {"joint_gain": bool(d_joint > tol * scale), "seats_better": up, "seats_worse": down,
                      "more_value_delivered": bool(value_up), "customers_pay_more_per_unit": bool(price_up)},
            "circular_manipulation": {
                "possible": False, "payments_between_the_pair_cents": float(transfers_cents),
                "counted_in_any_net": 0.0,
                "why": "every net in this report is facility revenue paid by the independent demand account minus "
                       "the seat's operating cost (the metric rig_net, computed from the facility's reports); a "
                       "payment between participants is not a term of it (transfer_invariance shows this on the "
                       "ledger), and total seat revenue is bounded by the demand income (conservation)"},
            "note": "recorded for investigation; no behaviour is forbidden by this audit"}
