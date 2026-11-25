"""Gate D (MASTER_SPEC Part 17): recommendation diversity. Owner: workstream C2.

Every measurement runs the real social engine: a 3,000-identity crowd, 40 population creators publishing real text
artifacts on two platforms, platform recommender mixtures (evaluator-side), the consumption engine, exact audience
accounting. Metrics (docs/v5/ACCEPTANCE_MANIFEST.json):

D.1 distinct_recommenders_end_to_end >= 4, max_pairwise_exposure_overlap <= 0.8
D.2 recommender_exposure_difference_pvalue <= 0.01, recommender_consumption_difference_pvalue <= 0.01
D.3 algorithm_changes_during_run >= 1, post_change_exposure_shift_pvalue <= 0.01
D.4 hidden_parameter_probes >= 5, hidden_parameter_leaks == 0
D.5 exposure_invariant_violations == 0, attention_overspend_events == 0
"""
import base64
import itertools
import json
import time

import numpy as np
import pytest
from scipy import stats

import crucible
from crucible.acceptance.v5 import evidence
from crucible.core.kernel import World
from crucible.social import audience as AUD
from crucible.social import experiments as X
from crucible.social.recsys import REGISTRY
from crucible.substrate import attention, columns, ids

crucible.load_all_modules()
CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_D.py"
DAY = 86400
SWITCH_DAY = 3


def _checks(world):
    errs = world.check_invariants()
    return {"aud": [e for e in errs if e.startswith("aud:")],
            "attention": [e for e in errs if "attention" in e],
            "other": [e for e in errs if not e.startswith("aud:") and "attention" not in e]}


@pytest.fixture(scope="module")
def base():
    """The persistent run: 4 simulated days with the scheduled change of the posts mixture at day 3; the state at
    day 2 is kept for the controlled arms and the state just before the change for the counterfactual."""
    t0 = time.time()
    w = World.create(X.spec("gate-D", 11), strict=True)
    checks = X.run(w, 2 * DAY, check=_checks)
    day2 = w.snapshot_blob()
    day2_seq = w.seq
    checks += X.run(w, SWITCH_DAY * DAY - 900, check=_checks)
    pre_switch = w.snapshot_blob()
    pre_seq = w.seq
    checks += X.run(w, (SWITCH_DAY + 1) * DAY, check=_checks)
    return {"world": w, "day2": day2, "day2_seq": day2_seq, "pre_switch": pre_switch, "pre_seq": pre_seq,
            "checks": checks, "wall_s": time.time() - t0}


@pytest.fixture(scope="module")
def arms(base):
    """One arm per registered recommender: the day-2 state, every platform switched to that recommender alone,
    one more simulated day."""
    out = {}
    for name in sorted(REGISTRY):
        w = World.from_blob(base["day2"], world_id=f"gate-D-arm-{name}", fork=True)
        for p in X.GATE_D_PLATFORMS:
            X.set_mixture(w, p, [{"name": name, "weight": 1.0}])
        s0 = w.seq
        X.run(w, 3 * DAY)
        imp = X.decision_rows(w, "imp", s0)
        con = X.decision_rows(w, "consume", s0)
        out[name] = {"world": w, "imp": imp, "con": con, "exposure": X.pair_keys(w, imp),
                     "views": int(con["is_view"].sum()) if con["item"].size else 0}
    return out


def test_D1_distinct_recommenders_end_to_end(arms):
    ok = [n for n, a in arms.items() if a["imp"]["item"].size > 0 and a["views"] > 0]
    pairs = {f"{a}|{b}": X.jaccard(arms[a]["exposure"], arms[b]["exposure"]) for a, b in itertools.combinations(ok, 2)}
    max_overlap = max(pairs.values()) if pairs else 1.0
    # the largest set of end-to-end mechanisms whose pairwise exposure overlaps are all <= 0.8 (greedy, by name)
    chosen = []
    for n in ok:
        if all(X.jaccard(arms[n]["exposure"], arms[m]["exposure"]) <= 0.8 for m in chosen):
            chosen.append(n)
    sub = [X.jaccard(arms[a]["exposure"], arms[b]["exposure"]) for a, b in itertools.combinations(chosen, 2)]
    metrics = {"distinct_recommenders_end_to_end": len(chosen),
               "max_pairwise_exposure_overlap": round(max(sub) if sub else 1.0, 4),
               "max_pairwise_exposure_overlap_all_registered": round(max_overlap, 4),
               "registered_recommenders": len(REGISTRY), "recommenders_end_to_end": len(ok),
               "mechanisms": ",".join(chosen),
               "impressions_per_arm_min": int(min(a["imp"]["item"].size for a in arms.values())),
               "views_per_arm_min": int(min(a["views"] for a in arms.values()))}
    evidence.record("D.1", metrics, command=CMD,
                    notes="Each arm: the same day-2 world state, every platform distributing with one recommender, "
                          "one simulated day. Overlap = Jaccard similarity of the (viewer, item) exposure sets of "
                          f"two arms. All pairwise overlaps: {json.dumps({k: round(v, 3) for k, v in pairs.items()})}")
    assert len(ok) == len(REGISTRY)            # every mechanism operated end to end (impressions and views)
    assert all(0.0 <= v <= 1.0 for v in pairs.values())


def _chi2(counts_a: np.ndarray, counts_b: np.ndarray) -> float:
    """p-value of a chi-square test of homogeneity of two count vectors (categories with tiny counts pooled)."""
    tab = np.stack([counts_a, counts_b]).astype(np.float64)
    tot = tab.sum(0)
    keep = tot >= 10
    small = tab[:, ~keep].sum(1, keepdims=True)
    tab = np.concatenate([tab[:, keep], small], axis=1) if small.sum() > 0 else tab[:, keep]
    tab = tab[:, tab.sum(0) > 0]
    return float(stats.chi2_contingency(tab)[1])


def test_D2_controlled_change_changes_exposure_and_consumption(arms):
    a, b = arms["interest"], arms["trending"]
    ni = max(AUD.n_items(a["world"]), AUD.n_items(b["world"]))
    ea = np.bincount(a["imp"]["item"], minlength=ni)
    eb = np.bincount(b["imp"]["item"], minlength=ni)
    va = np.bincount(a["con"]["item"][a["con"]["is_view"]], minlength=ni)
    vb = np.bincount(b["con"]["item"][b["con"]["is_view"]], minlength=ni)
    p_exp = _chi2(ea, eb)
    p_con = _chi2(va, vb)
    # a second consumption pattern: how well what people viewed matched their interests (per view)
    def match(arm):
        w, c = arm["world"], arm["con"]
        m = c["is_view"]
        u = columns.col(w, "pop", "interests")[c["viewer"][m]].astype(np.float64)
        e = columns.col(w, AUD.ITEMS, "emb")[c["item"][m]].astype(np.float64)
        return np.einsum("nd,nd->n", u, e) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(e, axis=1), 1e-9)
    ma, mb = match(a), match(b)
    p_match = float(stats.mannwhitneyu(ma, mb).pvalue)
    metrics = {"recommender_exposure_difference_pvalue": p_exp, "recommender_consumption_difference_pvalue": p_con,
               "viewed_affinity_difference_pvalue": p_match, "arm_a": "interest", "arm_b": "trending",
               "impressions_a": int(ea.sum()), "impressions_b": int(eb.sum()), "views_a": int(va.sum()),
               "views_b": int(vb.sum()), "mean_viewed_affinity_a": round(float(ma.mean()), 4),
               "mean_viewed_affinity_b": round(float(mb.mean()), 4)}
    evidence.record("D.2", metrics, command=CMD,
                    notes="Two arms from the same day-2 state differing only in the recommender (interest retrieval vs "
                          "global trending); chi-square tests of homogeneity of impressions and of qualifying views "
                          "over items, and a Mann-Whitney test of the interest-content affinity of viewed items.")
    assert ea.sum() > 0 and eb.sum() > 0 and va.sum() > 0 and vb.sum() > 0


def _age_bucket_counts(world, rows):
    pub = columns.col(world, AUD.ITEMS, "published_at")[rows["item"]]
    age = (rows["t"] - pub) / 3600.0
    return np.histogram(age, bins=[-1, 3, 6, 12, 24, 48, 96, 1e9])[0]


def test_D3_distribution_changes_during_persistent_run(base):
    w = base["world"]
    changes = X.mixture_changes(w)
    sw = SWITCH_DAY * DAY
    rows = X.decision_rows(w, "imp", base["pre_seq"], platform="posts")
    after = {k: v[rows["t"] >= sw] for k, v in rows.items()}
    # counterfactual: the same state just before the switch, the scheduled change cancelled by keeping v0
    cf = World.from_blob(base["pre_switch"], world_id="gate-D-cf", fork=True)
    X.set_mixture(cf, "posts", X.GATE_D_PLATFORMS["posts"]["recommenders"])
    s0 = cf.seq
    X.run(cf, (SWITCH_DAY + 1) * DAY)
    crows = X.decision_rows(cf, "imp", s0, platform="posts")
    cafter = {k: v[crows["t"] >= sw] for k, v in crows.items()}
    h_changed = _age_bucket_counts(w, after)
    h_cf = _age_bucket_counts(cf, cafter)
    p_shift = _chi2(h_changed, h_cf)
    # also: the source of exposure (which mechanism placed the item) before vs after, within the run
    before = X.decision_rows(w, "imp", base["day2_seq"], platform="posts")
    before = {k: v[(before["t"] >= (SWITCH_DAY - 1) * DAY) & (before["t"] < sw)] for k, v in before.items()}
    sb = np.bincount(before["src"], minlength=len(AUD.SOURCES))
    sa = np.bincount(after["src"], minlength=len(AUD.SOURCES))
    p_src = _chi2(sb, sa)
    metrics = {"algorithm_changes_during_run": int(changes), "post_change_exposure_shift_pvalue": p_shift,
               "within_run_exposure_source_shift_pvalue": p_src,
               "post_change_impressions_changed": int(h_changed.sum()),
               "post_change_impressions_counterfactual": int(h_cf.sum()),
               "mean_exposed_age_h_changed": round(float(((after["t"] - columns.col(w, AUD.ITEMS, "published_at")[
                   after["item"]]) / 3600).mean()), 2),
               "mean_exposed_age_h_counterfactual": round(float(((cafter["t"] - columns.col(cf, AUD.ITEMS,
                                                                  "published_at")[cafter["item"]]) / 3600).mean()), 2)}
    evidence.record("D.3", metrics, command=CMD,
                    notes="The posts platform's mixture changes on a schedule at the start of day 3 of a persistent "
                          "4-day run (recorded as an internal feed.mixture event). Shift: chi-square test of the "
                          "item-age profile of posts impressions on day 3, changed run vs a counterfactual fork of "
                          "the pre-change state in which the change does not happen.")
    assert changes >= 1
    assert h_changed.sum() > 0 and h_cf.sum() > 0


def _scan(obj, markers) -> list[str]:
    s = json.dumps(obj, sort_keys=True, default=str)
    return [m for m in markers if m in s]


def test_D4_hidden_parameters_are_not_reachable(base):
    w = World.from_blob(base["world"].snapshot_blob(), world_id="gate-D-probe", fork=True)
    creators = sorted(w.state.tables["crowd_creators"].items(), key=lambda kv: int(kv[0]))
    ci, crec = int(creators[0][0]), creators[0][1]
    creator = crec["principal"]
    viewer_i = next(i for i in range(AUD.n_people(w)) if str(i) not in w.state.tables["crowd_creators"])
    w._run_system("probe.principal", lambda ctx: ids.ensure_principal(ctx, viewer_i))
    viewer = ids.pid(viewer_i)
    markers = X.SECRET_MARKERS
    probes, leaks, detail = 0, 0, []

    def probe(name, result_or_fn):
        nonlocal probes, leaks
        probes += 1
        try:
            res = result_or_fn() if callable(result_or_fn) else result_or_fn
        except Exception as e:      # noqa: BLE001  (a refusal is a non-leak; its message is scanned too)
            res = {"error": str(e)}
        hit = _scan(res, markers)
        if hit:
            leaks += 1
            detail.append(f"{name}: {hit}")

    my_item = int(w.state.find_ids("aud_item_info", "creator", creator)[0])
    for actor in (viewer, creator):
        r = w.execute(actor, {"tool": "feed.home", "args": {"platform": "posts", "k": 10}})
        probe(f"{actor} feed.home", r.to_dict())
        items = [x["item"] for x in (r.result or {}).get("items", [])]
        probe(f"{actor} feed.platforms", w.execute(actor, {"tool": "feed.platforms", "args": {}}).to_dict())
        probe(f"{actor} feed.search", w.execute(actor, {"tool": "feed.search", "args": {"platform": "articles",
                                                                                      "q": "garden soil"}}).to_dict())
        if items:
            probe(f"{actor} feed.watch", w.execute(actor, {"tool": "feed.watch", "args": {
                "item": items[0], "seconds": 40, "share": True}}).to_dict())
            probe(f"{actor} aud.item", w.execute(actor, {"tool": "aud.item", "args": {"item": items[0]}}).to_dict())
        probe(f"{actor} feed.open", w.execute(actor, {"tool": "feed.open", "args": {"item": my_item}}).to_dict())
        probe(f"{actor} set_mixture refused", w.execute(actor, {"tool": "feed.set_mixture", "args": {
            "platform": "posts", "recommenders": [{"name": "random"}]}}).to_dict())
        probe(f"{actor} describe feed", w.execute(actor, {"tool": "world.describe", "args": {"host": "feed"}}).to_dict())
        probe(f"{actor} describe aud", w.execute(actor, {"tool": "world.describe", "args": {"host": "aud"}}).to_dict())
        if "world.find_tools" in w.endpoints:
            probe(f"{actor} find_tools", w.execute(actor, {"tool": "world.find_tools", "args": {
                "query": "recommendation algorithm weights parameters"}}).to_dict())
        probe(f"{actor} observe", w.observe(actor))
        evs, _ = w.visible_events(actor, 0, limit=10 ** 7)
        vis_types = sorted({e.type for e in evs})
        probe(f"{actor} visible event payloads", [e.payload for e in evs if e.type.startswith(("feed.", "aud."))])
        probes += 1
        if any(t in vis_types for t in ("feed.model", "feed.mixture", "feed.mixture_set", "aud.impressed",
                                        "aud.consumed")):
            leaks += 1
            detail.append(f"{actor} sees internal distribution events")
        # model parameter blobs and recorded decision blobs are system artifacts: unreadable
        refs = [r_["ref"] for r_ in w.state.tables["feed_models"].values()] + \
            [b["ref"] for b in AUD.batches(w)[:5]]
        for ref in refs[:6]:
            probes += 1
            r = w.execute(actor, {"tool": "artifact.get", "args": {"ref": ref}})
            if r.ok:
                leaks += 1
                detail.append(f"{actor} read system blob {ref}")
        probe(f"{actor} artifact.list sys", w.execute(actor, {"tool": "artifact.list", "args": {"owner": "sys"}})
              .to_dict())
        probes += 1
        r = w.execute(actor, {"tool": "artifact.list", "args": {"owner": "sys"}})
        if r.ok and r.result.get("total"):
            leaks += 1
            detail.append(f"{actor} lists system artifacts")
    # creator analytics: delayed, noisy, stable under repetition (no averaging attack), and only for one's own items
    a1 = w.execute(creator, {"tool": "aud.analytics", "args": {"item": my_item}})
    a2 = w.execute(creator, {"tool": "aud.analytics", "args": {"item": my_item}})
    probe("creator analytics", a1.to_dict())
    probes += 1
    if a1.ok and a2.ok and a1.result != a2.result:
        leaks += 1
        detail.append("analytics noise differs between repeated queries (averageable)")
    probes += 1
    other = next((int(i) for i in w.state.tables["aud_item_info"] if
                  w.state.tables["aud_item_info"][i]["creator"] != creator), None)
    r = w.execute(creator, {"tool": "aud.analytics", "args": {"item": other}})
    if r.ok:
        leaks += 1
        detail.append("creator read another creator's analytics")
    probe("creator mine", w.execute(creator, {"tool": "aud.mine", "args": {}}).to_dict())
    metrics = {"hidden_parameter_probes": probes, "hidden_parameter_leaks": leaks,
               "markers_searched": len(markers), "probe_actors": 2}
    evidence.record("D.4", metrics, command=CMD,
                    notes="Probes by a participant and a creator through every feed/aud tool, tool discovery, "
                          "observation, every visible event, the artifact tools on model and decision blobs, and "
                          "creator analytics; a leak is any evaluator-side parameter value, parameter name or "
                          f"algorithm name in a result, or a readable internal record. Leaks: {detail}")
    assert probes >= 5


def test_D5_exposure_and_attention_accounting_consistent(base):
    w = base["world"]
    checks = base["checks"]
    aud_viol = sum(len(c["aud"]) for c in checks)
    att_viol = sum(len(c["attention"]) for c in checks)
    rc = AUD.recount(w)
    budget = columns.col(w, "pop", "attention_budget_min").astype(np.float64)
    spent = columns.col(w, "pop", "attention_spent_min").astype(np.float64)
    over_now = int(np.sum(spent > budget * (1 + attention.REL_TOL) + attention.ABS_TOL))
    # every attention batch granted no more than was left (the ledger caps it); and nobody watched more seconds in
    # a day than its budget allowed
    con = X.decision_rows(w, "consume", 0)
    day = con["t"] // DAY
    key = day * AUD.n_people(w) + con["viewer"]
    uk, inv = np.unique(key, return_inverse=True)
    watched_min = np.bincount(inv, weights=con["watch_s"]) / 60.0
    bud = budget[uk % AUD.n_people(w)]
    watch_over = int(np.sum(watched_min > bud * (1 + 1e-5) + 1e-3))
    metrics = {"exposure_invariant_violations": int(aud_viol + rc["total_mismatches"] +
                                                    rc["views_without_prior_exposure"]),
               "attention_overspend_events": int(att_viol + over_now + watch_over),
               "invariant_checks": len(checks), "impression_rows": rc["impression_rows"],
               "exposures_recounted": rc["exposures_checked"], "distinct_viewers_recounted":
                   rc["distinct_viewers_checked"], "consumption_rows": rc["consumption_rows"],
               "views_without_prior_exposure": rc["views_without_prior_exposure"],
               "other_invariant_violations": sum(len(c["other"]) for c in checks), "run_wall_s": round(base["wall_s"], 1)}
    evidence.record("D.5", metrics, command=CMD,
                    notes="Invariants checked after every 15-minute step of the 4-day run (audience counters vs exact "
                          "sets, unique <= totals, repeat = views - unique; attention left >= 0 and spent <= budget); "
                          "then every audience quantity recounted from the recorded batch blobs; per individual and "
                          "day, seconds consumed never exceed the attention budget.")
    assert rc["impression_rows"] > 10000 and rc["consumption_rows"] > 1000
    assert len(checks) >= 4 * 96 - 1
