"""Gate B (MASTER_SPEC Part 17): persistent autonomous browsing. Owner: workstream B+C (CORE-POP).

ONE integrated world, the standard population profile (10,000 lightweight + 256 adaptive individuals of
`crucible.population`; strategic and semantic classes are measured in Gates G and H), for 30 simulated days: the 4.0
internet with generated web sites (popsites: rich, thin, broken, flaky, spam, off-topic), the population's web
browsing, search and forum (popweb), and the whole social internet on the same population (content catalogue, posts,
blog, communities, images, developer and knowledge platforms, product discovery, the consumption engine, exact
audience accounting, members who write and publish). Nothing is initiated by a learner: no agent acts at all.

At day 20 the world is forked into three controlled interventions (B.6): in each, two of the most visited useful
(rich) sites are redeployed as broken through ordinary tools (code.commit + cloud.deploy by their operator), against
an untouched control fork; two more days each.

Metrics (docs/v5/ACCEPTANCE_MANIFEST.json):
B.1 distinct_participants >= 10000
B.2 learner_initiated_consumer_events == 0, autonomous_sessions >= 100000
B.3 preference_effect_pvalue <= 0.01, preference_effect_size >= 0.05
B.4 return_gap_useful_vs_unsatisfactory >= 0.05, abandon_gap_unsatisfactory_vs_useful >= 0.05, p-values <= 0.01
B.5 simulated_days >= 30, days_with_activity >= 30, last_week_over_first_week_activity >= 0.5
B.6 intervention_effect_pvalue <= 0.01, intervention_seeds >= 3
"""
import copy
import math
import time

import numpy as np
import pytest
from scipy import stats

import crucible
from crucible.acceptance.v5 import evidence
from crucible.core.kernel import World
from crucible.population import analysis as A
from crucible.population import check_provenance
from crucible.population import sites as SITES
from crucible.social import audience as AUD
from crucible.social.worlds import social_world
from crucible.substrate import columns

crucible.load_all_modules()
CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_B.py"
DAY = 86400
DAYS = 30
CUT_DAY = 20
PLATFORMS = ["posts", "blog", "community", "pics", "dev", "kb", "launch"]
pytestmark = [pytest.mark.slow, pytest.mark.timeout(4 * 3600)]


def _spec(seed: int):
    return social_world("standard", seed, web=True, sites=16, adaptive=True,
                        classes={"adaptive": 256, "strategic": 0, "semantic": 0}, platforms=PLATFORMS,
                        params={"log_retain": 4_000_000})


def _day_events(w, s0):
    return [e for e in w.log if e.seq > s0]


def _concat(logs):
    keys = logs[0].keys()
    return {k: np.concatenate([lg[k] for lg in logs]) for k in keys}


@pytest.fixture(scope="module")
def run():
    t0 = time.time()
    w = World.create(_spec(5), strict=True)
    labels = {}
    sessions, visit_logs, invariant_errors, provenance = {}, [], [], []
    learner = {"population_decision_events": 0, "learner_initiated": 0}
    cut = None
    s0 = 0
    for d in range(DAYS):
        w.advance_time((d + 1) * DAY)
        ev = _day_events(w, s0)
        labels.update(A.site_labels(w, ev))
        for k, v in A.sessions_by_day(w, ev).items():
            sessions[k] = sessions.get(k, 0) + v
        visit_logs.append(A.visit_log(w, ev))
        li = A.learner_initiated(w, ev)
        for k in learner:
            learner[k] += li[k]
        provenance.append(check_provenance(w, ev))
        invariant_errors += w.check_invariants()
        s0 = w.seq
        if d + 1 == CUT_DAY:
            cut = {"blob": w.snapshot_blob(), "t": w.now, "log": _concat([lg for lg in visit_logs if len(lg["ind"])])}
    log = _concat([lg for lg in visit_logs if len(lg["ind"])])
    return {"world": w, "labels": labels, "sessions": sessions, "log": log, "invariants": invariant_errors,
            "provenance": provenance, "learner": learner, "cut": cut, "wall_s": time.time() - t0,
            "proc_failed": sum(1 for e in w.log if e.type == "proc.failed")}


def _rows(w, labels, kinds):
    hosts = A.res_hosts(w)
    return [hosts.index(h) for h, s in labels.items() if s["behaviour"] in kinds and h in hosts]


def test_B1_world_with_10000_distinct_participants(run):
    w = run["world"]
    n_sess = columns.col(w, "pop", "n_sessions")
    cls = columns.col(w, "pop", "class")
    metrics = {"distinct_participants": int(np.sum(n_sess > 0)), "population_rows": int(n_sess.size),
               "lightweight_active": int(np.sum((n_sess > 0) & (cls == 1))),
               "adaptive_active": int(np.sum((n_sess > 0) & (cls == 2))), "run_wall_s": round(run["wall_s"], 1)}
    evidence.record("B.1", metrics, command=CMD,
                    notes="Distinct individuals of the one population (crucible.population, standard profile without "
                          "the strategic and semantic classes) that took at least one autonomous session in the "
                          "integrated 30-day world.")
    assert metrics["population_rows"] >= 10_000


def test_B2_autonomous_browsing_without_a_learner(run):
    prov = run["provenance"]
    li = run["learner"]
    forced = sum(p["forced_rows"] for p in prov)
    participant = sum(p["participant_rows"] for p in prov)
    metrics = {"learner_initiated_consumer_events": int(li["learner_initiated"] + participant + forced),
               "autonomous_sessions": int(sum(run["sessions"].values())),
               "population_decision_events": int(li["population_decision_events"]),
               "audience_decision_rows_with_provenance": int(sum(p["organic_population_rows"] for p in prov)),
               "audience_rows_forced": int(forced), "web_visits": int(len(run["log"]["ind"])),
               "failed_processes": int(run["proc_failed"])}
    evidence.record("B.2", metrics, command=CMD,
                    notes="Every population decision event of the run is checked for an in-world actor's command "
                          "(population.analysis.learner_initiated) and every audience decision batch for decision "
                          "provenance (population.check_provenance: listed by the population's own pop.session batch "
                          "and inside its activated cohort). No learner or agent acts in this world.")
    assert run["proc_failed"] == 0


def test_B3_individual_preferences_affect_decisions(run):
    w = run["world"]
    pe = A.preference_effect(w, log=run["log"])
    metrics = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in pe.items()}
    evidence.record("B.3", metrics, command=CMD,
                    notes="For the same resource, first visitors whose interests (shared semantic space) matched its "
                          "content better returned more often: the pooled return-rate difference above vs below each "
                          "resource's median match, Cochran-Mantel-Haenszel test stratified by resource.")
    assert pe["pairs"] > 1000


def test_B4_return_to_useful_and_leave_unsatisfactory(run):
    w = run["world"]
    useful = _rows(w, run["labels"], {"rich"})
    bad = _rows(w, run["labels"], {"broken", "flaky", "spam"})
    ra = A.return_abandon(w, useful, bad, log=run["log"])
    metrics = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in ra.items()}
    metrics.update(useful_sites=len(useful), unsatisfactory_sites=len(bad))
    evidence.record("B.4", metrics, command=CMD,
                    notes="First (individual, site) visits followed for 7 days: return rate to useful (rich) sites vs "
                          "unsatisfactory ones (broken, flaky, spam; the generator's evaluator-only labels), and the "
                          "rate of abandoning them; two-proportion z-tests.")
    assert ra["pairs_useful"] > 100 and ra["pairs_unsatisfactory"] > 100


def test_B5_activity_persists_30_days(run):
    s = run["sessions"]
    days = sorted(s)
    first = sum(s.get(d, 0) for d in range(0, 7))
    last = sum(s.get(d, 0) for d in range(DAYS - 7, DAYS))
    metrics = {"simulated_days": int(run["world"].now // DAY), "days_with_activity": int(sum(1 for d in days if s[d] > 0)),
               "last_week_over_first_week_activity": round(last / max(first, 1), 4),
               "sessions_first_week": int(first), "sessions_last_week": int(last),
               "invariant_errors": len(run["invariants"]), "items_published": int(AUD.n_items(run["world"])),
               "communities": len(run["world"].state.tables["communities"])}
    evidence.record("B.5", metrics, command=CMD, notes="Sessions per simulated day of the one population over 30 days; "
                                                       f"world invariants checked daily ({len(run['invariants'])} "
                                                       "errors).")
    assert not run["invariants"], run["invariants"][:5]


def _intervene(f, hosts, labels, rng):
    def do(ctx):
        st = ctx.state
        for h in hosts:
            dep = st.get("deployments", st.get("hosts", h)["deployment"])
            person = st.get("orgs", dep["owner"])["founder"]
            acct = ctx.world.modules["bank"].primary_account(st, dep["owner"])
            sp = SITES.make_spec(rng, h, labels[h]["topic"], "broken", n_articles=6)
            ctx.world.modules["code"].commit_files(ctx, dep["repo"], SITES.app_files(sp, acct), "site update",
                                                   person, trusted=True)
            ctx.call("cloud.deploy", {"repo": dep["repo"], "host": h, "instance": dep["instance"]},
                     as_principal=person)
    f._run_system("gate-B.intervene", do)


def test_B6_controlled_resource_changes_change_behaviour(run):
    cut = run["cut"]
    labels = run["labels"]
    lg = cut["log"]
    base = World.from_blob(cut["blob"], world_id="gate-B-cut", fork=True)
    hosts = A.res_hosts(base)
    rich = [h for h, s in labels.items() if s["behaviour"] == "rich" and h in hosts]
    vis = {h: int((lg["res"] == hosts.index(h)).sum()) for h in rich}
    ranked = sorted(rich, key=lambda h: (-vis[h], h))
    reps, ps, gaps = [], [], []
    for r in range(3):
        treated = ranked[2 * r: 2 * r + 2]
        if len(treated) < 2:
            treated = ranked[:2]
        arms = {}
        for arm in ("control", "treatment"):
            f = World.from_blob(cut["blob"], world_id=f"gate-B-{r}-{arm}", fork=True)
            if arm == "treatment":
                _intervene(f, treated, labels, np.random.default_rng(100 + r))
            s0 = f.seq
            f.advance_time(cut["t"] + 2 * DAY)
            v = A.visit_log(f, [e for e in f.log if e.seq > s0])
            hs = A.res_hosts(f)
            rows = [hs.index(h) for h in treated]
            m = np.isin(v["res"], rows)
            arms[arm] = {"visits": int(len(v["ind"])), "visits_treated": int(m.sum()),
                         "abandons_treated": int(v["abandon"][m].sum()),
                         "mean_util_treated": float(v["util"][m].mean()) if m.any() else 0.0}
        c, t = arms["control"], arms["treatment"]
        gap, p = A.two_proportion(t["abandons_treated"], max(t["visits_treated"], 1), c["abandons_treated"],
                                  max(c["visits_treated"], 1))
        reps.append({"treated": treated, **{f"{a}_{k}": v for a in arms for k, v in arms[a].items()},
                     "abandon_rate_gap": round(gap, 4), "pvalue": p})
        ps.append(max(p, 1e-300))
        gaps.append(gap)
    fisher = float(stats.combine_pvalues(ps, method="fisher")[1])
    metrics = {"intervention_effect_pvalue": max(ps), "intervention_seeds": len(reps),
               "fisher_combined_pvalue": fisher, "mean_abandon_rate_gap": round(float(np.mean(gaps)), 4),
               "min_abandon_rate_gap": round(float(np.min(gaps)), 4)}
    evidence.record("B.6", metrics, command=CMD,
                    notes="Three controlled interventions on the day-20 state of the 10,256-individual world: each "
                          "redeploys two of the most visited rich sites (different pairs) as broken through their "
                          "operator's ordinary tools, against an untouched control fork, two more days. Effect: the "
                          "rate at which visits to the changed sites end in abandoning them, treatment vs control "
                          "(two-proportion z-test per replicate; the criterion is the largest of the three p-values). "
                          f"Replicates: {reps}")
    assert len(reps) == 3 and all(math.isfinite(g) for g in gaps)
