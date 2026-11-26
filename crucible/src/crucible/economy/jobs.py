"""The population in the labour market (activity `jobs`, run by `economy.work`; MASTER_SPEC Parts 6.6, 9.9).

D-15: employment in a v5 world is decided by the one population. An individual without an employer (the
population's own `employed` feature) looks for work in the world's labour market (`commerce.labor`): it registers
itself once (declared skill names evidenced by its own strongest skills; its reservation wage is private), reads the
open postings, applies to those its declared skills cover whose pay reaches its reservation, and answers the offers
it receives - accept at or above its reservation, counter at its reservation when the offer is within 20% of it
(three rounds at most), otherwise reject. Acceptance signs an employment contract (salary paid monthly from the
employer's account to the individual's own account by the contract; a missed salary is a breach that ends the job).

An employee works a shift on its working days (`labor.shift`: recorded hours, the employer's support queue). It
quits (terminates its contract with the contract's notice) when it finds an employer of its own outside this market
(the population's `employed` feature turns true) or when its salary has fallen below 80% of its reservation (an
employer's pay cut it accepted earlier cannot keep it).

The reservation wage is the individual's value of time over the job's hours, discounted while it is out of work:
`vot (cents/min) x 60 x hours_week x 4.33 x 0.6`, with `vot` from the individual's means (`economy.work`). These
rules are a documented, uncalibrated decision model (`work.jobs`, fidelity register FR-WORK-2).
"""
from __future__ import annotations

import numpy as np

from ..core.errors import CrucibleError
from ..substrate import columns, ids
from . import popbridge as pb

DAY = 86400
MODEL = {"model_id": "work.jobs", "model_version": "1", "fidelity_id": "FR-WORK-2"}
TABLE = "jobs"
DEFAULTS = {"search_days": 3, "hours_week": 20, "max_apply": 2, "reservation_factor": 0.6, "counter_band": 0.8,
            "quit_below": 0.8}


def declare(world, n: int) -> None:
    columns.declare(world, TABLE, n, {"last_search": ("int64", (), -1),
                                      "last_shift_day": ("int32", (), -1), "vot": ("float32", (), 0.0),
                                      "choice": ("int32", (), -1)})


def _cfg(work, k):
    return (work.config.get("jobs") or {}).get(k, DEFAULTS[k])


def _labor_skills(skills_row: np.ndarray) -> list[str]:
    from ..commerce.labor import POP_SKILLS
    from ..population.schema import SKILL_NAMES
    order = np.argsort(-np.asarray(skills_row, np.float64), kind="stable")[:2]
    out: list[str] = []
    for j in order.tolist():
        for s in POP_SKILLS.get(SKILL_NAMES[j], ()):
            if s not in out:
                out.append(s)
    return out[:5]


def _vot(work, ctx, i: int, wallet: float, employed: bool) -> float:
    """Value of a minute of the individual's time (cents): the work module's own value when the individual is a
    professional, else drawn once from its means exactly as `economy.work` draws it."""
    if TABLE in columns.tables(ctx.world) and float(columns.col(ctx.world, TABLE, "vot")[i]) > 0:
        return float(columns.col(ctx.world, TABLE, "vot")[i])
    if int(work._col("fam")[i]) >= 0:
        return float(work._col("vot")[i])
    rng = ctx.world.rng_for("jobs-vot", int(i))
    v = np.exp(rng.normal(np.log(max(wallet, 1000.0) / 60000.0 * 12.0), 0.35)) * (1.0 if employed else 0.6)
    return float(np.clip(v, 2.0, 120.0))


def session(work, ctx, idx: np.ndarray, now: int) -> np.ndarray:
    """The `jobs` activity: one session per due individual (see the module docstring)."""
    w, st = ctx.world, ctx.state
    util = np.zeros(idx.size)
    if "labor" not in w.modules or idx.size == 0:
        return util
    f = pb.features(w, idx)
    employed = f["employed"].astype(bool)
    hours = int(_cfg(work, "hours_week"))
    today = int(now // DAY)
    dec: dict[str, list] = {k: [] for k in ("register", "apply", "accept", "counter", "reject", "shift", "quit")}
    vot_set, searched = [], []
    for k, i in enumerate(idx.tolist()):
        p = ids.pid(i)
        rec = st.get("labor_workers", p)
        if rec is None:
            if employed[k]:
                continue
            vot = _vot(work, ctx, i, float(f["wallet_cents"][k]), False)
            res = int(vot * 60 * hours * 4.33 * float(_cfg(work, "reservation_factor")))
            pb.fund(ctx, i)
            try:
                ctx.call("labor.register", {"declared": _labor_skills(f["skills"][k]), "hours_week": hours,
                                            "reservation_cents_month": res}, as_principal=p)
            except CrucibleError:
                continue
            dec["register"].append(i)
            vot_set.append((i, vot))
            rec = st.get("labor_workers", p)
        res = int(rec["reservation"])
        if rec["employer"]:
            c = st.get("contracts", rec["contract"]) if rec["contract"] else None
            if c is None or c["status"] != "active":
                continue
            if employed[k] or int(rec["salary"]) < float(_cfg(work, "quit_below")) * res:
                try:
                    ctx.call("contracts.terminate", {"contract": rec["contract"]}, as_principal=p)
                    dec["quit"].append(i)
                    util[k] += 0.1
                except CrucibleError:
                    pass
                continue
            last = int(columns.col(w, TABLE, "last_shift_day")[i])
            if last < today and (today % 7) < 5:
                try:
                    r = ctx.call("labor.shift", {"org": rec["employer"], "hours": round(hours / 5.0, 2)},
                                 as_principal=p)
                    dec["shift"].append(i)
                    util[k] += 0.2 + 0.02 * int(r.get("tickets_resolved", 0))
                except CrucibleError:
                    pass
            continue
        # a job seeker: answer offers first
        answered = False
        for oid in st.find_ids("labor_offers", "worker", p):
            o = st.tables["labor_offers"][oid]
            if o["status"] != "open" or o["from"] != "org" or now > o["expires"]:
                continue
            if int(o["salary"]) >= res:
                action, sal = "accept", None
            elif int(o["salary"]) >= float(_cfg(work, "counter_band")) * res and int(o["round"]) < 3:
                action, sal = "counter", res
            else:
                action, sal = "reject", None
            args = {"offer": oid, "action": action, "message": "" if action == "accept" else f"I need {res}"}
            if sal is not None:
                args["salary_cents_month"] = int(sal)
            try:
                ctx.call("labor.respond", args, as_principal=p)
                dec[action].append(i)
                answered = True
                util[k] += 0.4 if action == "accept" else 0.0
            except CrucibleError:
                continue
            if action == "accept":
                break                                     # one job: other offers wait (they expire)
        if answered:
            continue
        last = int(columns.col(w, TABLE, "last_search")[i])
        if last >= 0 and now - last < int(_cfg(work, "search_days")) * DAY:
            continue
        mine = set(rec["declared"] or [])
        applied = 0
        for post in ctx.call("labor.postings", {"limit": 50}, as_principal=p):
            if applied >= int(_cfg(work, "max_apply")):
                break
            if st.has("labor_apps", f"{post['post']}|{p}") or post["org"] in (rec["blacklist"] or []):
                continue
            if not set(post["skills"]) <= mine:
                continue
            pay = int(post["salary_max"]) * min(1.0, hours / max(1, int(post["hours_week"])))
            if pay < res:
                continue
            try:
                ctx.call("labor.apply", {"post": post["post"], "note": "I would like to work for you."},
                         as_principal=p)
                applied += 1
            except CrucibleError:
                pass
        if applied:
            dec["apply"].append(i)
        searched.append(i)
    if searched:
        columns.batch(ctx, TABLE, "jobs.searched", np.array(searched),
                      {"last_search": np.full(len(searched), now, np.int64)}, mode="set")
    if dec["shift"]:
        columns.batch(ctx, TABLE, "jobs.shift_day", np.array(dec["shift"]),
                      {"last_shift_day": np.full(len(dec["shift"]), today, np.int32)}, mode="set")
    if vot_set:
        columns.batch(ctx, TABLE, "jobs.registered", np.array([i for i, _ in vot_set]),
                      {"vot": np.array([v for _, v in vot_set], np.float32)}, mode="set")
    for kind, lst in dec.items():
        if lst:
            u = np.array(sorted(set(lst)))
            pb.record_decisions(ctx, f"work.job_{kind}", u, {"choice": np.ones(u.size, np.int32)}, table=TABLE,
                                **MODEL)
    return util


def due(work, ctx) -> np.ndarray:
    """Individuals the daily process wakes for `jobs`: labour-market employees on working days, registered job
    seekers with open offers or a search due, and a 1/search_days share of the individuals without an employer
    (the population decides itself whether to look)."""
    w = ctx.world
    n = pb.n_individuals(w)
    st = ctx.state
    out = set()
    for p, r in st.scan("labor_workers"):
        i = ids.index(p)
        if i is not None:
            out.add(int(i))
    rng = ctx.rng("jobs.wake")
    k = max(1, n // int(_cfg(work, "search_days")))
    sample = np.unique(rng.choice(n, size=min(n, k), replace=False)) if n else np.zeros(0, np.int64)
    if sample.size:
        emp = pb.features(w, sample)["employed"].astype(bool)
        out.update(int(x) for x in sample[~emp].tolist())
    return np.array(sorted(out), np.int64)
