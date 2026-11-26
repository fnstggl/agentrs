"""Evaluator-side measurements of the population (never reachable by in-world actors).

Everything here is recomputed from the recorded log: the cohort batch events (`cols.batch` of tables `pop` and
`res`), the array blobs they reference (the visit log, the initial state) and the ordinary kernel events (forum,
bank). Nothing reads a counter that was not written by an applier of a recorded decision.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from ..substrate import artifacts, attention, columns
from . import schema

DAY = 86400
VISIT_KEYS = ("ind", "res", "via", "first", "util", "pages", "ok_pages", "abandon", "bought", "paid", "evaluated",
              "cls", "minutes", "match")


def _events(world, events=None) -> Iterable:
    return world.log if events is None else events


def visit_log(world, events=None) -> dict[str, np.ndarray]:
    """One row per visit: the arrays of every visit blob, plus `t` and `activity`."""
    parts, ts, acts = [], [], []
    for e in _events(world, events):
        if e.type != "cols.batch" or e.payload["table"] != schema.RES_TABLE:
            continue
        if not e.payload["kind"].startswith("popweb.visits."):
            continue
        ref = e.payload["extra"].get("visits_ref")
        if not ref:
            continue
        a = artifacts.get_array_blob(world, ref)
        parts.append(a)
        ts.append(np.full(len(a["ind"]), e.payload["extra"]["t"], np.int64))
        acts.append(np.full(len(a["ind"]), e.payload["kind"].rsplit(".", 1)[1]))
    if not parts:
        return {k: np.zeros(0) for k in VISIT_KEYS + ("t", "activity")}
    out = {k: np.concatenate([p[k] for p in parts]) for k in VISIT_KEYS}
    out["t"] = np.concatenate(ts)
    out["activity"] = np.concatenate(acts)
    return out


def site_labels(world, events=None) -> dict[str, dict]:
    """host -> {behaviour, topic, category, products} of the generated sites (from the evaluator-only note)."""
    out = {}
    for e in _events(world, events):
        if e.type == "note" and e.payload.get("what") == "popsites":
            for s in e.payload["sites"]:
                out[s["host"]] = s
    return out


def res_hosts(world) -> list[str]:
    tbl = world.state.tables["pop_res"]
    hosts = [""] * len(tbl)
    for h, r in tbl.items():
        hosts[r["row"]] = h
    return hosts


def initial(world, names: Iterable[str], events=None, strict: bool = True) -> dict[str, np.ndarray]:
    """The genesis values of `pop` columns (from the recorded `pop.init` batches)."""
    names = list(names)
    n = columns.rows(world, schema.TABLE)
    out = {}
    for e in _events(world, events):
        if e.type == "cols.batch" and e.payload["table"] == schema.TABLE and e.payload["kind"] == "pop.init":
            a = artifacts.get_array_blob(world, e.payload["ref"])
            ix = a["idx"].astype(np.int64)
            for k in names:
                if f"v.{k}" in a:
                    if k not in out:
                        col = columns.col(world, schema.TABLE, k)
                        out[k] = np.zeros(col.shape, col.dtype)
                    out[k][ix] = a[f"v.{k}"]
    missing = [k for k in names if k not in out]
    if missing and strict:
        raise KeyError(f"no recorded initial values for {missing} (derived initialisation?)")
    return out


def batches(world, table: str, kind_prefix: str, events=None):
    for e in _events(world, events):
        if e.type == "cols.batch" and e.payload["table"] == table and e.payload["kind"].startswith(kind_prefix):
            yield e


def sessions_by_day(world, events=None) -> dict[int, int]:
    out: dict[int, int] = {}
    for e in batches(world, schema.TABLE, "pop.session", events):
        d = int(e.payload["extra"]["t"]) // DAY
        out[d] = out.get(d, 0) + int(e.payload["extra"]["sessions"])
    return out


def session_individuals(world, events=None) -> np.ndarray:
    """Distinct individuals that took at least one session (row indices): the takers of the population's sessions
    (`pop.learn` batches) and the rows a class runtime activated (`pop.session` batches of classes 3-4)."""
    seen = set()
    for e in _events(world, events):
        if e.type != "cols.batch" or e.payload["table"] != schema.TABLE:
            continue
        k = e.payload["kind"]
        if k == "pop.learn" or (k == "pop.session" and (e.payload.get("extra") or {}).get("class")):
            seen.update(artifacts.get_array_blob(world, e.payload["ref"])["idx"].tolist())
    return np.array(sorted(seen), np.int64)


POP_EVENT_PREFIXES = ("pop.", "popweb.", "pop_adaptive.", "pop_strategic.")


def learner_initiated(world, events=None) -> dict:
    """Population decisions recorded inside a command of an in-world actor (rather than the population's own
    processes): {population_decision_events, learner_initiated}."""
    total = by_cmd = 0
    for e in _events(world, events):
        pop_ev = (e.type == "cols.batch" and e.payload["table"] in (schema.TABLE, schema.RES_TABLE)) or \
            e.type.startswith(POP_EVENT_PREFIXES)
        if not pop_ev:
            continue
        total += 1
        if e.cmd is not None and e.actor != "sys":
            by_cmd += 1
    return {"population_decision_events": total, "learner_initiated": by_cmd}


def two_proportion(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """(p1 - p2, two-sided p-value) of a two-proportion z-test."""
    from scipy.stats import norm
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(max(p * (1 - p) * (1 / n1 + 1 / n2), 1e-300))
    z = (p1 - p2) / se
    return p1 - p2, float(2 * norm.sf(abs(z)))


def pair_outcomes(world, window_s: int = 7 * DAY, events=None, log=None) -> dict[str, np.ndarray]:
    """For every (individual, resource) first visit early enough to be followed for `window_s`: did the individual
    return within the window, and did it abandon the resource within the window."""
    v = visit_log(world, events) if log is None else log
    if not len(v["ind"]):
        return {k: np.zeros(0) for k in ("ind", "res", "t0", "returned", "abandoned", "first_util", "first_match")}
    end = world.now
    key = v["ind"].astype(np.int64) * 1_000_003 + v["res"].astype(np.int64)
    order = np.lexsort((v["t"], key))
    key, t, first, ab = key[order], v["t"][order], v["first"][order], v["abandon"][order]
    util, mt = v["util"][order], v["match"][order]
    ind, res = v["ind"][order], v["res"][order]
    out = {k: [] for k in ("ind", "res", "t0", "returned", "abandoned", "first_util", "first_match")}
    starts = np.flatnonzero(first)
    for s in starts:
        t0 = t[s]
        if t0 + window_s > end:
            continue
        j = s + 1
        ret = abn = False
        abn = bool(ab[s])
        while j < len(key) and key[j] == key[s] and not first[j]:
            if t[j] <= t0 + window_s:
                ret = True
                abn = abn or bool(ab[j])
            j += 1
        out["ind"].append(ind[s])
        out["res"].append(res[s])
        out["t0"].append(t0)
        out["returned"].append(ret)
        out["abandoned"].append(abn)
        out["first_util"].append(util[s])
        out["first_match"].append(mt[s])
    return {k: np.array(x) for k, x in out.items()}


def return_abandon(world, useful_rows, bad_rows, window_s: int = 7 * DAY, events=None, log=None) -> dict:
    po = pair_outcomes(world, window_s, events, log)
    u = np.isin(po["res"], list(useful_rows))
    b = np.isin(po["res"], list(bad_rows))
    ru, nu = int(po["returned"][u].sum()), int(u.sum())
    rb, nb = int(po["returned"][b].sum()), int(b.sum())
    au, ab = int(po["abandoned"][u].sum()), int(po["abandoned"][b].sum())
    gap_r, p_r = two_proportion(ru, nu, rb, nb)
    gap_a, p_a = two_proportion(ab, nb, au, nu)
    return {"pairs_useful": nu, "pairs_unsatisfactory": nb, "return_rate_useful": ru / max(nu, 1),
            "return_rate_unsatisfactory": rb / max(nb, 1), "abandon_rate_useful": au / max(nu, 1),
            "abandon_rate_unsatisfactory": ab / max(nb, 1), "return_gap_useful_vs_unsatisfactory": gap_r,
            "return_pvalue": p_r, "abandon_gap_unsatisfactory_vs_useful": gap_a, "abandon_pvalue": p_a}


def preference_effect(world, window_s: int = 7 * DAY, events=None, log=None) -> dict:
    """Do individual preferences affect decisions? For the SAME resource, individuals whose interests (as they were
    when they first visited it) match its content better should come back more often. Within each resource the
    first-visit pairs are split at that resource's median match; the effect is the pooled difference in return rate
    (above - below the median) and the test a Cochran-Mantel-Haenszel test stratified by resource (so differences
    between resources cannot produce it)."""
    from scipy.stats import chi2
    po = pair_outcomes(world, window_s, events, log)
    if not len(po["ind"]):
        return {"preference_effect_size": 0.0, "preference_effect_pvalue": 1.0, "pairs": 0}
    match = po["first_match"].astype(np.float64)
    num = var = 0.0
    k_hi = n_hi = k_lo = n_lo = 0
    strata = 0
    for r in np.unique(po["res"]):
        m = po["res"] == r
        if m.sum() < 10:
            continue
        med = np.median(match[m])
        hi = m & (match > med)
        lo = m & (match <= med)
        a = int(po["returned"][hi].sum())
        b = int(hi.sum()) - a
        c_ = int(po["returned"][lo].sum())
        d = int(lo.sum()) - c_
        n = a + b + c_ + d
        if n < 2 or (a + b) == 0 or (c_ + d) == 0 or (a + c_) == 0 or (b + d) == 0:
            continue
        strata += 1
        num += a - (a + b) * (a + c_) / n
        var += (a + b) * (c_ + d) * (a + c_) * (b + d) / (n * n * max(n - 1, 1))
        k_hi, n_hi, k_lo, n_lo = k_hi + a, n_hi + a + b, k_lo + c_, n_lo + c_ + d
    stat = (abs(num) - 0.5) ** 2 / var if var > 0 else 0.0
    p = float(chi2.sf(stat, 1)) if var > 0 else 1.0
    eff = k_hi / max(n_hi, 1) - k_lo / max(n_lo, 1)
    return {"preference_effect_size": float(eff), "preference_effect_pvalue": p, "pairs": int(n_hi + n_lo),
            "resources": strata, "return_rate_high_match": k_hi / max(n_hi, 1),
            "return_rate_low_match": k_lo / max(n_lo, 1),
            "match_return_correlation": float(np.corrcoef(match, po["returned"].astype(float))[0, 1])
            if len(match) > 2 and np.std(match) > 0 else 0.0}


def same_situation_heterogeneity(world, events=None, log=None, min_group: int = 5) -> dict:
    """Individuals in the same situation - the same resource, read in the same tick, so the same recorded page
    responses - respond differently: the mean within-group standard deviation of the experienced utility and of
    the number of pages read (a decision), and how much of the utility variation individual interests explain."""
    v = visit_log(world, events) if log is None else log
    if not len(v["ind"]):
        return {"same_situation_response_heterogeneity": 0.0, "groups": 0}
    key = v["res"].astype(np.int64) * 10**12 + v["t"].astype(np.int64)
    u_keys, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    big = np.flatnonzero(cnt >= min_group)
    sd_u, sd_p, r2 = [], [], []
    cur = columns.col(world, schema.TABLE, "interests").astype(np.float32)
    rv = columns.col(world, schema.RES_TABLE, "vec").astype(np.float32)
    from .text import cosine
    for g in big[:2000]:
        m = inv == g
        if v["ok_pages"][m].min() == 0:
            continue
        sd_u.append(float(np.std(v["util"][m])))
        sd_p.append(float(np.std(v["pages"][m])))
        x = cosine(cur[v["ind"][m].astype(np.int64)], rv[v["res"][m].astype(np.int64)])
        y = v["util"][m].astype(float)
        if np.std(x) > 0 and np.std(y) > 0:
            r2.append(float(np.corrcoef(x, y)[0, 1]) ** 2)
    return {"same_situation_response_heterogeneity": float(np.mean(sd_u)) if sd_u else 0.0,
            "same_situation_pages_sd": float(np.mean(sd_p)) if sd_p else 0.0,
            "same_situation_groups": len(sd_u),
            "utility_variance_explained_by_interest_match": float(np.mean(r2)) if r2 else 0.0}


def preference_drift(world, events=None, log=None) -> dict:
    """Preferences evolve through experience: the change of each individual's interests since genesis points towards
    the content it enjoyed. drift = mean cosine(interests_now - interests_0, enjoyed_content - interests_0) over
    individuals with enjoyed visits; individuals without any visit keep their interests (control)."""
    v = visit_log(world, events) if log is None else log
    init = initial(world, ["interests"], events)["interests"].astype(np.float32)
    cur = columns.col(world, schema.TABLE, "interests").astype(np.float32)
    rv = columns.col(world, schema.RES_TABLE, "vec").astype(np.float32)
    n = init.shape[0]
    acc = np.zeros_like(init)
    good = v["util"] > 0 if len(v["ind"]) else np.zeros(0, bool)
    np.add.at(acc, v["ind"][good].astype(np.int64), rv[v["res"][good].astype(np.int64)] *
              v["util"][good][:, None].astype(np.float32))
    has = np.linalg.norm(acc, axis=1) > 0
    d0 = acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-9) - init
    dlt = cur - init
    from .text import cosine
    cs = cosine(dlt[has], d0[has])
    visited = np.zeros(n, bool)
    visited[v["ind"].astype(np.int64)] = True
    still = np.linalg.norm(dlt[~visited], axis=1) if (~visited).any() else np.zeros(0)
    return {"preference_drift_from_experience": float(np.mean(cs)) if cs.size else 0.0,
            "individuals_with_drift": int(has.sum()),
            "mean_interest_change": float(np.linalg.norm(dlt[has], axis=1).mean()) if has.any() else 0.0,
            "unvisited_mean_interest_change": float(still.mean()) if still.size else 0.0}


def state_fields(world, genesis: dict | None = None, events=None) -> dict:
    """For each Part 3.1 field: represented (columns exist), heterogeneous (differs between individuals now) and
    evolving (differs now from its value right after genesis for some individual). `genesis` holds the columns as
    they were when the world was created (a test copies them); without it the recorded `pop.init` values (or the
    declared fill values) are the reference."""
    spec = columns.tables(world)[schema.TABLE]["columns"]
    names = sorted({c for cs in schema.FIELDS.values() for c in cs if c in spec})
    if genesis is None:
        genesis = initial(world, names, events, strict=False)
        for c in names:
            if c not in genesis:
                dt, shape, fill = spec[c]
                genesis[c] = np.full(columns.col(world, schema.TABLE, c).shape, fill, np.dtype(dt))
    rep = {}
    for f, cs in schema.FIELDS.items():
        present = [c for c in cs if c in spec]
        het = evo = False
        for c in present:
            a = columns.col(world, schema.TABLE, c)
            flat = a.reshape(a.shape[0], -1)
            if flat.shape[0] > 1 and np.any(flat != flat[0]):
                het = True
            if c in genesis and not np.array_equal(genesis[c], a):
                evo = True
        rep[f] = {"columns": present, "represented": bool(present), "heterogeneous": het, "evolving": evo}
    return rep


def behaviours(world, events=None, log=None, minimum: int = 10) -> dict:
    """Counts of the 13 browsing behaviours of Part 4.1 in the recorded log."""
    v = visit_log(world, events) if log is None else log
    ev = list(_events(world, events))
    searches = sum(int(e.payload["n"]) for e in ev if e.type == "cols.batch" and e.payload["kind"] == "popweb.searched")
    reads = sum(int(e.payload["extra"].get("read", 0)) for e in ev if e.type == "cols.batch" and
                e.payload["kind"] == "popweb.forum")
    pop_actor = lambda a: isinstance(a, str) and a[:1] == "p" and a[1:].isdigit()  # noqa: E731
    follows = [e for e in ev if e.type == "forum.follow" and pop_actor(e.payload["follower"])]
    switches = sum(int(e.payload["extra"].get("switches", 0)) for e in ev if e.type == "cols.batch" and
                   e.payload["kind"] == "pop.session")
    counts = {
        "discover_websites": int(v["first"].sum()) if len(v["ind"]) else 0,
        "search_for_information": searches,
        "browse_feeds": reads,
        "visit_pages": int(v["pages"].sum()) if len(v["ind"]) else 0,
        "consume_media": int((v["minutes"] >= 1.0).sum()) + reads if len(v["ind"]) else reads,
        "follow_creators": sum(1 for e in follows if not e.payload["target"].startswith("board:")),
        "join_communities": sum(1 for e in follows if e.payload["target"].startswith("board:")),
        "participate_in_discussions": sum(1 for e in ev if e.type in ("forum.reply", "forum.post") and
                                          pop_actor(e.payload["author"])),
        "evaluate_products": int(v["evaluated"].sum()) if len(v["ind"]) else 0,
        "purchase_services": int(v["bought"].sum()) if len(v["ind"]) else 0,
        "return_to_useful_resources": int(((~v["first"]) & (v["util"] > 0.3)).sum()) if len(v["ind"]) else 0,
        "leave_resources": int(v["abandon"].sum()) if len(v["ind"]) else 0,
        "move_between_platforms": switches,
    }
    viol = attention.check(world)
    return {"counts": counts, "present": sorted(k for k, c in counts.items() if c >= minimum),
            "browsing_behaviours": sum(1 for c in counts.values() if c >= minimum),
            "attention_violations": len(viol)}


def social_behaviours(world, events=None) -> dict:
    """Counts of the Part 4.1 behaviours the one population performs on the social internet (recorded decision
    batches and events of the social modules; evaluator side)."""
    ev = list(_events(world, events))
    pop_actor = lambda a: isinstance(a, str) and a[:1] == "p" and a[1:].isdigit()  # noqa: E731
    out = {"browse_feeds": 0, "consume_media": 0, "search_for_information": 0, "follow_creators": 0,
           "join_communities": 0, "participate_in_discussions": 0, "leave_resources": 0, "share_content": 0,
           "subscribe": 0, "evaluate_products": 0, "purchase_services": 0}
    for e in ev:
        if e.type != "cols.batch":
            if e.type == "content.published" and pop_actor(e.payload["creator"]) and e.payload.get("community"):
                out["participate_in_discussions"] += 1
            elif e.type == "content.comment" and pop_actor(e.payload["author"]):
                out["participate_in_discussions"] += 1
            elif e.type in ("prod.purchase", "market.order") and pop_actor(e.payload.get("buyer")):
                out["purchase_services"] += 1
            continue
        p = e.payload
        k = p["kind"]
        if p["table"] == schema.TABLE and k.startswith("impress."):
            out["browse_feeds"] += int(p["n"])
            src = artifacts.get_array_blob(world, p["ref"])["v.aud_i_src"]
            from ..social.audience import SOURCE_CODE
            out["search_for_information"] += int(np.sum(np.isin(src, [SOURCE_CODE["search"],
                                                                      SOURCE_CODE["search_tool"]])))
        elif p["table"] == schema.TABLE and k.startswith("consume."):
            out["consume_media"] += int(p["n"])
        elif p["table"] == "follow" and k == "social.follow" and (p.get("extra") or {}).get("decision"):
            out["follow_creators"] += int(p["n"])
        elif p["table"] == "follow" and k == "social.unfollow":
            out["leave_resources"] += int(p["n"])
        elif p["table"] == "share" and (p.get("extra") or {}).get("decision"):
            out["share_content"] += int(p["n"])
        elif p["table"] == "cmember" and k == "community.join":
            out["join_communities"] += int(p["n"])
        elif p["table"] == "cmember" and k == "community.leave":
            out["leave_resources"] += int(p["n"])
        elif p["table"] == "nlsub" and k == "letters.subscribe":
            out["subscribe"] += int(p["n"])
        elif p["table"] == schema.TABLE and k == "pop_social.shop":
            out["evaluate_products"] += int((p.get("extra") or {}).get("visits", 0))
    return out
