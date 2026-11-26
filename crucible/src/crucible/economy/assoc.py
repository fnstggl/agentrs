"""The population in institutions: mutual-aid associations (activity `assoc`, run by `economy.work`; Part 6.6).

D-15: who founds, joins, pays into, votes in and leaves an institution is decided by the one population, in its own
sessions, through the institutions' ordinary tools (`commerce.institutions`: `inst.found`, `inst.join`,
`inst.pay_dues`, `inst.propose`, `inst.vote`; identity `id.leave`). Membership lives in the identity registry's
membership table (the one owner), dues in the institution's contribution record, benefits only in bank moves the
members decided.

A mutual-aid association insures its members against losing their job: members pay monthly dues; a member who
loses its work (`popbridge.employment`: the population's own employer, or an employment contract of the labour
market - a member dismissed by its labour-market employer claims like any other) claims the published benefit by
proposing a transfer from the association's account to itself; the charter puts every transfer to a members'
majority vote with a quorum. Each individual decides from its own state:

* **found** - an employed, sociable individual (sociability >= `found_sociability`) that belongs to no
  association and finds none with room founds one (at most one founding a day per world), with dues of about 2% of
  its own monthly means and a benefit of `benefit_months` months of dues; the charter is the standard one below;
* **join** - an employed individual weighs the insurance: it joins an open association when its risk aversion
  (1 - error tolerance) times its belief of losing its job within a month (the share of its contacts out of work,
  shrunk to a 5% prior) times the benefit exceeds the dues, and the dues are at most 3% of its monthly means;
* **pay** - each month a member pays its dues if it can afford three months of them; otherwise it falls behind;
* **claim** - a member that lost its job since its last session and has not claimed in this spell proposes its
  benefit (a member in arrears may still try);
* **vote** - on a claim: yes when the claimant paid dues at least twice in the last 90 days (the association's
  members-only dues record) and the amount is within the published benefit, with probability 0.9 (no otherwise,
  with probability 0.9); on a rule change: yes with probability 0.6;
* **adapt** - when two of the association's last three decisions lapsed without a quorum, a member proposes
  lowering the quorum of the members' procedure by 0.1 (not below 0.1): an `amend` decision under the charter's own
  rules (institutions adapt their operational rules through authorised mechanisms);
* **leave** - a member leaves when its claim was rejected, or when its dues exceed 5% of its monthly means.

These are documented, uncalibrated decision rules (`work.assoc`, fidelity register FR-WORK-3).
"""
from __future__ import annotations

import numpy as np

from ..core.errors import CrucibleError
from ..substrate import columns, ids
from . import popbridge as pb

DAY = 86400
MONTH = 30 * DAY
MODEL = {"model_id": "work.assoc", "model_version": "1", "fidelity_id": "FR-WORK-3"}
TABLE = "assoc"
DEFAULTS = {"found_sociability": 0.85, "benefit_months": 12, "max_members": 400, "consider_days": 30,
            "quorum": 0.3, "window_days": 3}
CHARTER = lambda quorum, window: {  # noqa: E731
    "procedures": {"members": {"type": "majority", "electorate": "members", "quorum": quorum,
                               "window_s": int(window) * DAY}},
    "rules": [{"match": {"kind": "tool", "tool": "bank.*"}, "procedure": "members"},
              {"match": {"kind": "amend"}, "procedure": "members"}],
    "default": "members", "membership": {"admission": "open", "max_members": 400},
    "dissolution": {"distribution": "contribution"}}


def declare(world, n: int) -> None:
    columns.declare(world, TABLE, n, {"dues_month": ("int32", (), -1), "was_employed": ("int8", (), -1),
                                      "claimed": ("bool", (), False), "last_consider": ("int64", (), -1),
                                      "choice": ("int32", (), -1)})


def _cfg(work, k):
    return (work.config.get("assoc") or {}).get(k, DEFAULTS[k])


def associations(st) -> list[tuple[str, dict]]:
    """Active associations with published mutual-aid terms (public)."""
    out = []
    for oid, o in st.scan("orgs"):
        terms = (o.get("info") or {}).get("terms") or {}
        if o["status"] == "active" and o["kind"] == "association" and terms.get("benefit_cents"):
            out.append((oid, terms))
    return sorted(out)


def _member_of(st, p) -> list[str]:
    return sorted(m["org"] for m in st.find("members", "person", p)
                  if (st.get("orgs", m["org"]) or {}).get("kind") == "association")


def _monthly_means(wallet: float) -> float:
    return max(1000.0, float(wallet) / 3.0)


def session(work, ctx, idx: np.ndarray, now: int) -> np.ndarray:
    w, st = ctx.world, ctx.state
    util = np.zeros(idx.size)
    if "inst" not in w.modules or idx.size == 0:
        return util
    f = pb.features(w, idx)
    hired = pb.labour_market_employees(w)
    emp = pb.employment(w, idx, hired)
    prefs = f["prefs"].astype(np.float64)
    contacts = f["contacts"]
    month = int(now // MONTH)
    rng = ctx.rng("assoc.session")
    dec: dict[str, list] = {k: [] for k in ("found", "join", "pay", "behind", "claim", "vote", "adapt", "leave")}
    founded_today = any(o.get("created", -1) // DAY == now // DAY for _, o in st.scan("orgs")
                        if o["kind"] == "association")
    col_emp = columns.col(w, TABLE, "was_employed")
    col_claimed = columns.col(w, TABLE, "claimed")
    upd_emp, upd_claim, upd_dues, upd_cons = {}, {}, {}, {}
    for k, i in enumerate(idx.tolist()):
        p = ids.pid(i)
        means = _monthly_means(float(f["wallet_cents"][k]))
        mine = _member_of(st, p)
        if mine:
            org = mine[0]
            terms = dict(st.tables["orgs"][org]["info"].get("terms") or {})
            dues = int(terms.get("dues_cents_month", 0))
            # votes on open decisions
            for did in st.find_ids("inst_decisions", "org", org):
                d = st.tables["inst_decisions"][did]
                if d["status"] != "open" or st.has("inst_ballots", f"{did}|{p}") or \
                        p not in {x for x, _ in d["electorate"]}:
                    continue
                yes = _vote(ctx, st, org, p, d, terms, rng)
                try:
                    ctx.call("inst.vote", {"decision": did, "choice": "yes" if yes else "no"}, as_principal=p)
                    dec["vote"].append(i)
                except CrucibleError:
                    pass
            # a rejected claim of its own: leave
            rejected = any(d["action"].get("kind") == "tool" and d["by"] == p and d["status"] == "rejected"
                           and now - int(d.get("decided_t") or now) < 7 * DAY
                           for d in st.find("inst_decisions", "org", org))
            if rejected or dues > 0.05 * means:
                try:
                    ctx.call("id.leave", {"org": org}, as_principal=p)
                    dec["leave"].append(i)
                except CrucibleError:
                    pass
                continue
            # dues
            if dues and int(columns.col(w, TABLE, "dues_month")[i]) < month:
                if float(f["wallet_cents"][k]) >= 3 * dues:
                    pb.fund(ctx, i)
                    try:
                        ctx.call("inst.pay_dues", {"org": org, "amount": dues}, as_principal=p)
                        dec["pay"].append(i)
                        upd_dues[i] = month
                    except CrucibleError:
                        dec["behind"].append(i)
                else:
                    dec["behind"].append(i)
            # claim after losing a job
            lost = int(col_emp[i]) == 1 and not emp[k]
            if lost and not bool(col_claimed[i]):
                bank = w.modules["bank"]
                pb.fund(ctx, i)
                try:
                    ctx.call("inst.propose", {"org": org, "action": {
                        "kind": "tool", "tool": "bank.transfer",
                        "args": {"from_account": bank.primary_account(st, org), "to": p,
                                 "amount": int(terms["benefit_cents"]), "memo": "mutual aid: job loss"}},
                        "reason": "I lost my job; I claim the association's benefit."}, as_principal=p)
                    dec["claim"].append(i)
                    upd_claim[i] = True
                except CrucibleError:
                    pass
            if emp[k] and bool(col_claimed[i]):
                upd_claim[i] = False
            # adapt the rules when decisions keep lapsing
            last3 = sorted((d for d in st.find("inst_decisions", "org", org) if d["status"] != "open"),
                           key=lambda d: int(d.get("decided_t") or 0))[-3:]
            if sum(1 for d in last3 if d["status"] == "lapsed") >= 2 and not any(
                    d["status"] == "open" and d["action"]["kind"] == "amend" for d in st.find("inst_decisions", "org", org)):
                c = st.tables["inst_charters"][org]["charter"]
                q = float(c["procedures"]["members"]["quorum"])
                if q > 0.1:
                    new = CHARTER(round(max(0.1, q - 0.1), 2), c["procedures"]["members"]["window_s"] // DAY)
                    new["membership"] = dict(c["membership"])
                    try:
                        ctx.call("inst.propose", {"org": org, "action": {"kind": "amend", "charter": new},
                                                  "reason": "Too few of us vote: lower the quorum."}, as_principal=p)
                        dec["adapt"].append(i)
                    except CrucibleError:
                        pass
        elif emp[k]:
            last = int(columns.col(w, TABLE, "last_consider")[i])
            if last >= 0 and now - last < int(_cfg(work, "consider_days")) * DAY:
                upd_emp[i] = 1
                continue
            upd_cons[i] = now
            open_ = [(o, t) for o, t in associations(st)
                     if len(st.find_ids("members", "org", o)) < int(_cfg(work, "max_members"))]
            risk = 1.0 - float(prefs[k, 2])
            cs = [int(c) for c in contacts[k] if int(c) >= 0]
            out_of_work = 0 if not cs else int((~pb.employment(w, np.array(cs), hired)).sum())
            p_loss = (out_of_work + 0.05 * 4) / (len(cs) + 4)
            best = None
            for o, t in open_:
                d_, b_ = int(t.get("dues_cents_month", 0)), int(t.get("benefit_cents", 0))
                if d_ > 0.03 * means:
                    continue
                v = risk * p_loss * b_ - d_
                if v > 0 and (best is None or v > best[0]):
                    best = (v, o)
            if best is not None:
                try:
                    pb.fund(ctx, i)
                    ctx.call("inst.join", {"org": best[1], "note": "I want to be insured against losing my job."},
                             as_principal=p)
                    dec["join"].append(i)
                except CrucibleError:
                    pass
            elif not open_ or all(len(st.find_ids("members", "org", o)) >= int(_cfg(work, "max_members"))
                                  for o, _ in open_):
                if float(prefs[k, 6]) >= float(_cfg(work, "found_sociability")) and not founded_today:
                    dues = int(max(300, round(0.02 * means, -2)))
                    try:
                        pb.fund(ctx, i)
                        ctx.call("inst.found", {
                            "handle": f"mutual-{i}", "name": f"Mutual Aid {i}", "kind": "association",
                            "charter": CHARTER(float(_cfg(work, "quorum")), int(_cfg(work, "window_days"))),
                            "terms": {"dues_cents_month": dues,
                                      "benefit_cents": dues * int(_cfg(work, "benefit_months")),
                                      "purpose": "members insure each other against losing their job"}},
                            as_principal=p)
                        dec["found"].append(i)
                        founded_today = True
                    except CrucibleError:
                        pass
        upd_emp[i] = 1 if emp[k] else 0
    for col, upd, dt in (("was_employed", upd_emp, np.int8), ("claimed", upd_claim, bool),
                         ("dues_month", upd_dues, np.int32), ("last_consider", upd_cons, np.int64)):
        if upd:
            ks = np.array(sorted(upd), np.int64)
            columns.batch(ctx, TABLE, f"assoc.{col}", ks, {col: np.array([upd[x] for x in ks.tolist()], dt)},
                          mode="set")
    for kind, lst in dec.items():
        if lst:
            u = np.array(sorted(set(lst)))
            pb.record_decisions(ctx, f"work.assoc_{kind}", u, {"choice": np.ones(u.size, np.int32)}, table=TABLE,
                                **MODEL)
            util[np.isin(idx, u)] += 0.05
    return util


def _vote(ctx, st, org, p, d, terms, rng) -> bool:
    a = d["action"]
    if a.get("kind") == "tool" and a.get("tool") == "bank.transfer":
        args = a.get("args") or {}
        to = args.get("to")
        if to == p:
            return True
        record = ctx.call("inst.dues", {"org": org, "days": 90}, as_principal=p)
        good = int((record.get(d["by"]) or {}).get("payments", 0)) >= 2 and \
            int(args.get("amount", 0)) <= int(terms.get("benefit_cents", 0))
        return bool(rng.random() < (0.9 if good else 0.1))
    return bool(rng.random() < 0.6)


def due(work, ctx) -> np.ndarray:
    """Individuals woken for `assoc` today: members (dues on the first days of a month, claims after a job loss,
    open decisions they have not voted on) and a 1/consider_days share of the employed individuals."""
    w, st = ctx.world, ctx.state
    n = pb.n_individuals(w)
    out = set()
    first_days = (ctx.now % MONTH) < 3 * DAY
    for m in st.scan("members"):
        rec = m[1]
        i = ids.index(rec["person"])
        if i is None or (st.get("orgs", rec["org"]) or {}).get("kind") != "association":
            continue
        out.add(int(i)) if first_days else None
    for did in st.find_ids("inst_decisions", "status", "open"):
        d = st.tables["inst_decisions"][did]
        for x, _ in d["electorate"]:
            i = ids.index(x)
            if i is not None and not st.has("inst_ballots", f"{did}|{x}"):
                out.add(int(i))
    members = [ids.index(rec["person"]) for _, rec in st.scan("members")
               if ids.index(rec["person"]) is not None and (st.get("orgs", rec["org"]) or {}).get("kind") == "association"]
    if members:
        mi = np.array(sorted(set(members)), np.int64)
        hired = pb.labour_market_employees(w)
        emp = pb.employment(w, mi, hired)
        was = columns.col(w, TABLE, "was_employed")[mi]
        out.update(int(x) for x in mi[(was == 1) & ~emp].tolist())
    rng = ctx.rng("assoc.wake")
    k = max(1, n // int(_cfg(work, "consider_days")))
    sample = np.unique(rng.choice(n, size=min(n, k), replace=False)) if n else np.zeros(0, np.int64)
    if sample.size:
        emp = pb.employment(w, sample, pb.labour_market_employees(w))
        out.update(int(x) for x in sample[emp].tolist())
    return np.array(sorted(out), np.int64)
