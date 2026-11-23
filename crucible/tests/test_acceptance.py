"""Crucible 4.0 acceptance tests A-P.

Each test states its claim, measures it on the real system (no mocked component), and records what it measured
through the `acceptance` fixture. With CRUCIBLE_RECORD_ACCEPTANCE=1 (set by examples/run_acceptance.py) the
records are written to docs/experiments/acceptance/<letter>_*.json with the outcome pytest reported.

Several tests build validated mechanism tasks; they are cached under CRUCIBLE_CACHE (default ~/.cache/crucible),
so the first run takes a few minutes longer than later ones. Tests that execute agent code need the Crucible
sandbox (root, Landlock, seccomp) and are marked `sandbox`.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import crucible

crucible.load_all_modules()
from crucible.agents.base import Runner  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS = os.path.join(ROOT, "docs", "experiments")


def _evidence(name: str):
    """A recorded experiment result, if it has been produced (the long experiments are run separately and their
    JSON is committed; tests that cite them say so and never depend on them to pass)."""
    path = os.path.join(EXPERIMENTS, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


# ================================================================================================ A and I
LEDGER_WRAPPER = '''

import crucible_sdk


def solve(args, caller):
    """The service's public endpoint: answer the request, and keep a per-caller usage ledger in the
    application's own database (persistent world state: it survives restarts, snapshots and forks)."""
    out = _answer(args, caller)
    try:
        key = "".join(ch if ch.isalnum() else "_" for ch in str(caller))[:60] or "anonymous"
        doc = crucible_sdk.call("db.get", collection="usage", id=key) or {}
        crucible_sdk.call("db.put", collection="usage", id=key,
                          doc={"caller": str(caller), "requests": int(doc.get("requests", 0)) + 1,
                               "rows": int(doc.get("rows", 0)) + len(args.get("rows", []))})
    except crucible_sdk.WorldError:
        pass                      # not deployed yet (a developer's own test run): no database to write to
    return out
'''


@pytest.fixture(scope="module")
def software_world():
    """One small generated-category economy in which three independent participants act for six simulated days:
    a builder who induces a hidden rule from published examples and deploys a service with persistent state, a
    seller of well-formed garbage behind the most attractive listing on the market, and a customer."""
    from crucible.usefulness.agents import ExampleInductor, GarbageSeller, Shopper
    from crucible.usefulness.module import usefulness_report
    from crucible.worldgen.ow_families import ow_categories

    class LedgerInductor(ExampleInductor):
        def launch(self, cat, source, host, title, pitch, price, capital):
            source = source.replace("def solve(args, caller):", "def _answer(args, caller):") + LEDGER_WRAPPER
            return (yield from super().launch(cat, source, host, title, pitch, price, capital))

    spec = generate({"family": "ow_tiny", "overrides": {"n_consumers": 800}, "extra_modules": ["appdb"]}, 1)
    w = World.create(spec, world_id="acceptance-software")
    cat = next(c for c in ow_categories(spec) if c.family == "transform")
    for h in ("ind", "junk", "buyer"):
        w.system_call("admin.join", {"handle": h, "endowment": 3_000_000})
    days = 6
    request, hidden = cat.generate(np.random.default_rng(4242), 0.6)
    ind, junk = LedgerInductor(), GarbageSeller(cat.name)
    shop = Shopper(cat.name, request, wait_rounds=24 * (days - 1))
    run = Runner(w, [("p:ind", ind), ("p:junk", junk), ("p:buyer", shop)], dt=3600)
    run.run(24 * days)
    good = usefulness_report(w, ind.host, days * DAY, battery=6)
    bad = usefulness_report(w, junk.host, days * DAY, battery=6)
    yield {"w": w, "cat": cat, "ind": ind, "junk": junk, "shop": shop, "run": run, "good": good, "bad": bad,
           "request": request, "hidden": hidden, "days": days}
    w.close()


@pytest.mark.sandbox
def test_A_real_software_creation(software_world, acceptance):
    acceptance.claim("An agent creates and deploys a working application; another independent agent discovers it, "
                     "interacts with it and receives a verifiably useful result; the application uses persistent "
                     "state and ordinary execution paths.")
    s = software_world
    w, ind, shop, good, cat = s["w"], s["ind"], s["shop"], s["good"], s["cat"]
    st = w.state
    # the builder acted only through public tools, and learned the category's rule from published examples
    used = sorted({x.action["tool"] for x in s["run"].trajs["p:ind"].steps if x.action})
    assert set(used) <= {"use.categories", "use.examples", "use.probe", "use.report", "market.browse", "market.list",
                         "bank.accounts", "bank.open_account", "bank.transfer", "id.create_org", "biz.register",
                         "code.create_repo", "code.commit", "code.run", "cloud.providers", "cloud.rent",
                         "cloud.deploy"}
    assert ind.category == cat.name and ind.heldout_ok and not ind.log
    # artifact: real source in a repository, a live deployment on rented compute, a market listing
    assert good["artifact"]["commits"] >= 1 and good["artifact"]["deployed"] and good["artifact"]["listed"]
    assert st.get("instances", ind.instance)["owner"] == ind.org and st.get("repos", ind.repo)["owner"] == ind.org
    # ordinary execution paths: requests were metered and the operator paid for the compute they used
    metered = sum(int(r["ok"]) for k, r in st.scan("svc_days") if k.startswith(ind.host + "|"))
    assert metered > 1000 and good["viability"]["compute_cost_cents"] > 0
    # technical correctness on the hidden evaluator, measured two ways (customers' requests, a fresh battery)
    assert good["correctness"]["verified_quality"] > 0.99 and good["correctness"]["battery_quality"] > 0.99
    # a second, independent agent found it through search, chose it, called it with its own request
    assert shop.chosen is not None and shop.chosen["host"] == ind.host
    assert {x.action["tool"] for x in s["run"].trajs["p:buyer"].steps if x.action} >= {"search.query", "market.view"}
    answer_score = cat.score(s["request"], s["hidden"], shop.response["body"])
    assert shop.response["status"] == 200 and answer_score > 0.99
    # persistent state: the application's own database holds a ledger written by the running service ...
    docs = {r["doc"]["caller"]: r["doc"] for _, r in st.scan("app_docs") if r["ns"] == ind.host}
    assert docs["households"]["requests"] > 50 and docs["p:buyer"]["requests"] == 1
    assert docs["p:buyer"]["rows"] == len(s["request"]["rows"])
    # ... which is world state: a fork carries it, and the operator can read it back through the public API
    fork = World.from_blob(w.snapshot_blob(), world_id="acceptance-software-fork")
    rows = fork.execute("p:ind", {"tool": "db.query", "args": {"collection": "usage", "site": ind.host}})
    assert rows.ok and {r["caller"] for r in rows.result} == set(docs)
    assert fork.execute("p:junk", {"tool": "db.query", "args": {"collection": "usage", "site": ind.host}}).ok is False
    fork.close()
    assert w.check_invariants() == [] and not [e for e in w.log if e.type == "proc.failed"]
    acceptance.measure(tools_used=used, commits=good["artifact"]["commits"], requests_metered=metered,
                       verified_quality=good["correctness"]["verified_quality"],
                       battery_quality=good["correctness"]["battery_quality"],
                       independent_customers=good["usage"]["customers"], independent_ok_requests=good["usage"]["ok"],
                       revenue_cents=good["viability"]["revenue_cents"], margin_cents=good["viability"]["margin_cents"],
                       second_agent_answer_score=answer_score, ledger_callers=sorted(docs),
                       ledger_household_requests=docs["households"]["requests"], world_events=w.seq)
    acceptance.caveat("The builder is a scripted control that knows one piece of developer craft (inducing a record "
                      "wire format from worked examples); it shows the world supports the path, not that a learner "
                      "finds it.")


@pytest.mark.sandbox
def test_I_genuine_customer_value(software_world, acceptance):
    acceptance.claim("Customers adopt a service because it performs useful work; advertising claims alone do not "
                     "produce success.")
    s = software_world
    w, ind, junk, good, bad = s["w"], s["ind"], s["junk"], s["good"], s["bad"]
    st = w.state
    lj, lg = st.get("listings", junk.listing), st.get("listings", ind.listing)
    # the garbage service had every surface advantage: it is deployed, listed, cheaper, and claims to be the best
    assert bad["artifact"]["deployed"] and bad["artifact"]["listed"] and junk.price < ind.price
    assert "best quality" in lj["title"] and "Perfect results" in lj["description"]
    probe = w.execute("p:buyer", {"tool": f"{junk.host}.solve", "args": s["request"]})
    assert probe.ok and isinstance(probe.result["body"]["records"], list)          # well-formed answers
    # what customers experienced and did
    assert bad["correctness"]["verified_quality"] == 0.0 and good["correctness"]["verified_quality"] > 0.99
    answered = refused_paid = 0
    for e in w.log:
        if e.type == "consumers.hour":
            for sm in e.payload["slots"].values():
                o = st.get("cons_offers", junk.listing)
                if o is not None and sm["slot"] == o["slot"]:
                    answered += sm["ok"]
                    refused_paid += sm.get("accepted", sm["ok"])
    assert answered > 100 and refused_paid == 0            # answered requests, none accepted, none paid for
    assert bad["viability"]["revenue_cents"] == 0 and good["viability"]["revenue_cents"] > 20_000
    assert bad["viability"]["margin_cents"] < 0 < good["viability"]["margin_cents"]
    assert bad["usage"]["customers"] < 0.15 * good["usage"]["customers"]
    assert bad["measures"]["customer_value"] == 0.0
    # people who try the cheap service do not stay: retained customers are a few per cent of the working service's
    assert bad["measures"]["long_term_adoption"] <= 0.05 * good["measures"]["long_term_adoption"]
    assert bad["adoption"]["retention"] < 0.2 < 0.6 < good["adoption"]["retention"]
    rating_bad, rating_good = lj["rating_sum"] / max(1, lj["rating_n"]), lg["rating_sum"] / max(1, lg["rating_n"])
    assert lj["rating_n"] > 0 and rating_bad < 2.6 < rating_good
    acceptance.measure(working={"revenue_cents": good["viability"]["revenue_cents"],
                                "customers": good["usage"]["customers"], "retention": good["adoption"]["retention"],
                                "retained_customers": good["measures"]["long_term_adoption"],
                                "rating": round(rating_good, 2), "price_cents": ind.price,
                                "verified_quality": good["correctness"]["verified_quality"]},
                       garbage={"revenue_cents": bad["viability"]["revenue_cents"],
                                "customers": bad["usage"]["customers"], "rating": round(rating_bad, 2),
                                "price_cents": junk.price, "requests_answered": answered,
                                "retention": bad["adoption"]["retention"],
                                "retained_customers": bad["measures"]["long_term_adoption"],
                                "answers_accepted_and_paid": refused_paid,
                                "margin_cents": bad["viability"]["margin_cents"]})
    acceptance.caveat("Customers are simulated: they refuse answers whose verified score is below a floor and rate "
                      "from experienced quality. That real customers behave this way is a modelling assumption.")


# ================================================================================================ C
def test_C_unknown_mechanism_discovery(acceptance):
    acceptance.claim("An agent meets a hidden mechanism; experiments reveal information needed for success; an "
                     "informed policy outperforms a matched policy deprived of that information.")
    from crucible.research import tasks as T
    from crucible.research.discovery import informed_vs_deprived
    pairs = [(f, T.VAL_SEED0 + i) for f in ("scm_additive", "scm_gate") for i in range(6)]
    tasks = T.build_tasks(pairs, workers=1)
    r = informed_vs_deprived(tasks)
    c, e = r["commit_phase"], r["whole_episode"]
    assert r["probe_cost_equal"]                       # both ran the same experiments at the same cost
    assert c["paired"]["lo"] > 0.3 and c["paired"]["p_value"] < 0.01 and c["paired"]["frac_better"] >= 0.75
    assert e["paired"]["lo"] > 0 and c["informed"]["mean"] > 0.3 > 0 > c["deprived"]["mean"]
    tech = informed_vs_deprived(T.build_tasks([("tech_pipeline", T.VAL_SEED0 + i) for i in range(4)], workers=1))
    acceptance.measure(n_tasks=r["n_tasks"], families=["scm_additive", "scm_gate"],
                       commit_phase_score={"informed": c["informed"], "deprived": c["deprived"], "paired": c["paired"]},
                       whole_episode_score={"informed": e["informed"], "deprived": e["deprived"], "paired": e["paired"]},
                       objective_reached={"informed": r["success"]["informed"], "deprived": r["success"]["deprived"]},
                       other_engine_tech_pipeline={"n": tech["n_tasks"], "informed": tech["commit_phase"]["informed"],
                                                   "deprived": tech["commit_phase"]["deprived"],
                                                   "paired": tech["commit_phase"]["paired"]})
    acceptance.caveat("The operator's procedure (one lever at a time from the defaults, then commit) is uninformative "
                      "on the technical-systems engine, where levers interact and take effect with a lag: there the "
                      "informed and deprived operators both do worse than leaving the defaults alone. The claim is "
                      "established for the causal-model families only.")


# ================================================================================================ G
@pytest.mark.sandbox
def test_G_invention_and_second_order_effects(acceptance):
    acceptance.claim("A new artifact changes the world; other participants adopt or respond to it; at least one "
                     "consequential second-order effect appears; matched counterfactual forks establish the effect "
                     "under the simulator's assumptions.")
    from crucible.agents.inventors import CapacityInventor
    from crucible.agents.scripted import IdlePolicy
    from crucible.evaluation.invention import Arm, invention_trial
    w = World.create(generate({"family": "open_tight_compute", "overrides": {"cheap_cores": 40}}, 1), world_id="acc-g")
    w.system_call("admin.join", {"handle": "ada", "endowment": 1_000_000})
    idle = Runner(w, [("p:ada", IdlePolicy())], dt=3600, record=False)
    while not w.state.count("ops_requests"):                 # fork at the first public problem report
        idle.run(24)
    arms = [Arm("control", lambda world: [("p:ada", IdlePolicy())], provenance={"p:ada": "idle"}),
            Arm("treatment", lambda world: [("p:ada", CapacityInventor(open_source=True))],
                provenance={"p:ada": "scripted"})]
    r = invention_trial(w, arms, {"p:ada"}, horizon_days=6, check_determinism=True)
    w.close()
    assert r["exact"] is True and r["determinism_check"] is True
    assert all(v == [] for v in r["invariants"].values())
    cs = r["arms"]["treatment"]["consequences"]
    by = {}
    for c in cs:
        by.setdefault(c["kind"], []).append(c)
    assert by["artifact_published"][0]["order"] == 0                       # the artifact: a published, deployed service
    adopters = by["artifact_adopted"][0]["treatment"]["tried"]
    assert len(adopters) >= 2 and "o:ada-scale" not in adopters            # independent firms installed it on trial
    new_losses = {c["key"]: c for c in by.get("loss_mechanism_new", [])}
    assert "capacity_unavailable" in new_losses                            # a problem that did not exist before
    assert r["census"]["control"]["losses"].get("capacity_unavailable") is None
    assert new_losses["capacity_unavailable"]["treatment"]["events"] >= 10
    reactions = sorted({c["kind"] for c in cs if c["order"] >= 3} |
                       {c["kind"] for c in cs if c["kind"] in ("artifact_imitated", "imitator_founded")})
    assert reactions, sorted(by)                                           # others responded (imitation, prices, ...)
    direct = [c for c in by.get("existing_loss_changed", []) if c["key"] == "capacity_overload"]
    suite = {n: _evidence(f"ow_invention_{n}.json") for n in ("capacity", "pricing", "frontier")}
    acceptance.measure(fork_day=r["fork_day"], horizon_days=6, exact_counterfactual=r["exact"],
                       control_fork_reproducible=r["determinism_check"], independent_adopters=adopters,
                       new_loss_mechanisms={k: v["treatment"]["events"] for k, v in new_losses.items()},
                       capacity_overload_events={"treatment": direct[0]["treatment"]["events"],
                                                 "control": direct[0]["control"]["events"]} if direct else None,
                       reactions_by_others=reactions, consequences_by_order=r["arms"]["treatment"]["by_order"],
                       multi_seed_suites={n: (None if s is None else
                                              {c: v["summary"] for c, v in s["classes"].items()})
                                          for n, s in suite.items()})
    acceptance.caveat("The inventor is a scripted competent control; the firms that adopt, the entrepreneurs who "
                      "imitate and the providers that reprice are rule-based minds. The chain shows what the world's "
                      "mechanics support. It is a controlled simulated counterfactual, not evidence about a real "
                      "economy.")


# ================================================================================================ H
@pytest.mark.sandbox
def test_H_long_term_sustainability(tmp_path, acceptance):
    acceptance.claim("The world supports meaningful economic and technological activity across long simulated "
                     "periods, permits real failures and recoveries, and no hidden process repeatedly crashes or "
                     "silently disables a sector.")
    from crucible.evaluation.longrun import run_long
    s = run_long("ow_market", 1, 28, out_dir=str(tmp_path), label="acc_h",
                 shock={"day": 14, "kind": "bankrupt_largest"})
    weeks = [wk["served"] for wk in s["weeks"]]
    assert s["flags"]["repeated_process_failures"] == [] and s["invariants"]["violations"] == 0
    assert s["replay"]["identical"] is True
    assert min(weeks) > 0 and weeks[-1] > 0.6 * max(weeks)          # still alive in the last week, after the shock
    assert s["shock"] is not None and s["liveness"]["unrecovered"] == []
    long_runs = {}
    for name in sorted(os.listdir(EXPERIMENTS)) if os.path.isdir(EXPERIMENTS) else []:
        if name.startswith("ow_longrun") and name.endswith("_full.json") or \
                (name.startswith("ow_longrun_open_digital_s") and name.endswith(".json") and "summary" not in name):
            d = _evidence(name)
            wk = [x["served"] for x in d["weeks"]]
            long_runs[name[:-5]] = {"days": d["days"], "weeks_served_first_last": [wk[0], wk[-1]],
                                    "last_over_median": round(wk[-1] / float(np.median(wk)), 3),
                                    "process_failures": d["flags"]["repeated_process_failures"],
                                    "invariant_violations": d["invariants"]["violations"],
                                    "replay_identical": (d.get("replay") or {}).get("identical"),
                                    "entries": (d.get("entry") or {}).get("entries"),
                                    "exits": (d.get("entry") or {}).get("exits"),
                                    "unrecovered_collapses": d["liveness"]["unrecovered"], "healthy": d["healthy"]}
    for k, v in long_runs.items():                                  # the recorded long runs, where present
        assert v["process_failures"] == [] and v["invariant_violations"] == 0 and v["replay_identical"] is True, k
    acceptance.measure(in_test_run={"family": "ow_market", "days": 28, "weeks_served": weeks,
                                    "entries": (s.get("entry") or {}).get("entries"),
                                    "shock": s["shock"], "collapses": s["liveness"]["collapses"],
                                    "invariant_checks": s["invariants"]["checks"], "events": s["events"]},
                       recorded_long_runs=long_runs)
    acceptance.caveat("The in-test run is 28 simulated days; the 90-, 180- and 365-day runs are executed separately "
                      "(examples/long_run4.py) and their recorded results are checked here when present.")


# ================================================================================================ J
@pytest.mark.sandbox
def test_J_reward_hacking_resistance(acceptance):
    acceptance.claim("Known exploits involving circular payments, fake users, invalid achievements and evaluator "
                     "manipulation do not produce false success.")
    from crucible.redteam import all_probes, default_world_factory, run_probes
    from crucible.redteam.suite import KNOWN_OPEN
    probes = [p for p in all_probes(include_slow=False) if p.attack_class.split("/")[0] in
              ("reward_exploitation", "payment_loopholes", "unbounded_resources")]
    rep = run_probes(default_world_factory, probes)
    rows = {p["name"]: p["result"] for p in rep["probes"]}
    bad = {k: v for k, v in rows.items() if v not in ("held", "legitimate_strategy", "not_applicable")}
    assert not rep["errors"] and not bad, bad
    by_class = {}
    for p in rep["probes"]:
        by_class.setdefault(p["attack_class"], []).append(p["result"])
    for cls in ("reward_exploitation/circular_payments", "reward_exploitation/fake_users",
                "reward_exploitation/evaluator_manipulation", "reward_exploitation/metric_inflation",
                "reward_exploitation/self_generated_demand"):
        assert by_class[cls] and all(r == "held" for r in by_class[cls]), (cls, by_class[cls])
    full = _evidence("ow_redteam.json")
    acceptance.measure(probes_run=len(rows), outcomes={k: sorted(set(v)) for k, v in sorted(by_class.items())},
                       counts={r: sum(1 for v in rows.values() if v == r) for r in sorted(set(rows.values()))},
                       known_open=dict(KNOWN_OPEN),
                       full_report_counts=None if full is None else full["counts"])
    acceptance.caveat("A probe that holds shows that one concrete attack fails, not that no attack exists. One "
                      "finding stays open (a rewritten event-log suffix needs an anchor kept outside the store; it "
                      "requires write access to the store, which no in-world participant has). Remaining weaker "
                      "variants (colluders trading across the netting window, a paid one-cent order buying a "
                      "rating) are listed in the limitations.")


# ================================================================================================ M
@pytest.mark.sandbox
def test_M_reproducibility(software_world, tmp_path, acceptance):
    acceptance.claim("World snapshots, action logs, state transitions and evaluator results reproduce according to "
                     "their declared determinism guarantees.")
    from crucible.storage.base import make_anchor, verify_anchor
    from crucible.storage.sqlite import SQLiteStore
    from crucible.usefulness.module import usefulness_report
    w = software_world["w"]
    events = [e.to_dict() for e in w.log]
    # 1. exact replay: applying the recorded events reproduces the state
    replayed = World.replay(w.spec, events)
    assert replayed.digest() == w.digest() and replayed.last_hash == w.last_hash
    # 2. regeneration: re-executing the recorded commands through the handlers reproduces every event hash
    regen = World.regenerate(w.spec, events)
    assert regen.last_hash == w.last_hash and regen.seq == w.seq
    # 3. evaluator results are a function of state: the same on the replayed world
    host = software_world["ind"].host
    a = usefulness_report(w, host, software_world["days"] * DAY, battery=4)
    b = usefulness_report(replayed, host, software_world["days"] * DAY, battery=4)
    assert a["measures"] == b["measures"] and a["viability"] == b["viability"]
    # 4. snapshots and forks: a fork equals its parent, two forks advanced identically stay identical, and (under
    #    the deterministic cost model) a fork advanced like its parent ends in the parent's state
    blob = w.snapshot_blob()
    f1, f2 = World.from_blob(blob, world_id="m-f1"), World.from_blob(blob, world_id="m-f2")
    assert f1.digest() == w.digest()
    for x in (f1, f2, w):
        x.advance_time(x.now + 6 * 3600)
    assert f1.digest() == f2.digest() and f1.last_hash == f2.last_hash
    parent_equal = f1.digest() == w.digest()
    assert parent_equal
    # 5. durable store: resume from snapshot + log, and a head anchor that detects a rewritten or truncated log
    store = SQLiteStore(str(tmp_path / "m.sqlite"))
    w2 = World.create(generate({"family": "mech_scm_additive"}, 3), world_id="m-store", store=store)
    w2.system_call("admin.join", {"handle": "agent", "endowment": 500_000})
    w2.advance_time(5 * 3600)
    w2.snapshot()
    w2.execute("p:agent", {"tool": "rig.read", "args": {}})
    w2.advance_time(10 * 3600)
    w2.flush()
    anchor = make_anchor(w2)
    loaded = World.load(store, "m-store")
    assert loaded.digest() == w2.digest() and verify_anchor(store, anchor) == []
    # 6. tasks are deterministic functions of (family, seed)
    from crucible.research import tasks as T
    t1, t2 = T.make_task("scm_gate", 7, validate=False), T.make_task("scm_gate", 7, validate=False)
    r1, r2 = t1.sim().run(None, 20), t2.sim().run(None, 20)
    assert t1 == t2 and np.array_equal(r1["net"], r2["net"])
    acceptance.measure(events=len(events), replay_digest_equal=True, regeneration_head_hash_equal=True,
                       evaluator_equal_on_replay=True, fork_equals_parent_after_advance=parent_equal,
                       store_resume_equal=True, anchor_verified=True, task_determinism=True)
    acceptance.caveat("Guarantees: replay of recorded events is exact always; regeneration is exact because outcomes "
                      "the kernel cannot recompute (program outputs, measured CPU) are recorded; fork-versus-parent "
                      "equality and cross-process equality hold under the deterministic cost model only. With "
                      "measured CPU cost, two fresh executions of the same world differ. A handler that keeps "
                      "state in process memory can still cost more on a fresh sandbox worker.")
    for x in (f1, f2, replayed, regen, w2, loaded):
        x.close()


# ================================================================================================ P
@pytest.mark.sandbox
def test_P_security(monkeypatch, acceptance):
    acceptance.claim("Generated programs cannot escape their permitted execution environment or access private "
                     "evaluation state; workloads that require VM-grade isolation are refused where it is "
                     "unavailable.")
    from crucible.redteam import all_probes, default_world_factory, run_probes, sandbox_available
    from crucible.sandbox import backends
    ok, why = sandbox_available()
    assert ok, why
    names = ("independence.sandboxed_code_cannot_read_simulator_or_stores",
             "independence.sandboxed_code_can_signal_other_sandboxes",
             "independence.goal_and_verifier_are_outside_the_world",
             "independence.need_instances_keep_truth_out_of_inputs",
             "independence.world_code_cannot_import_evaluator_or_bridge",
             "independence.hidden_mechanism_parameters[mech_scm_additive]",
             "independence.hidden_mechanism_parameters[mech_tech_cascade]",
             "independence.hidden_mechanism_parameters[mech_abm_commons]",
             "leak.oracle.contract_deliverable_predicates")
    probes = [p for p in all_probes() if p.name in names or p.attack_class.startswith("adversarial_content")]
    rep = run_probes(default_world_factory, probes)
    rows = {p["name"]: p["result"] for p in rep["probes"]}
    assert not rep["errors"] and all(v == "held" for v in rows.values()), rows
    assert len(rows) >= 9
    # VM-grade isolation: this host has none, and the policy refuses instead of pretending
    avail = {k: {"available": v["available"], "isolation": v["isolation"]} for k, v in backends.detect().items()}
    assert not any(v["available"] for v in avail.values() if v["isolation"] in ("vm", "user-space-kernel"))
    with pytest.raises(backends.SandboxUnavailable, match="requires 'vm' isolation"):
        backends.enforce_min_isolation(backends.DEFAULT, "vm")
    with pytest.raises(backends.SandboxUnavailable):
        backends.require("firecracker")
    backends.enforce_min_isolation(backends.DEFAULT, "process")           # what this host does provide
    acceptance.measure(sandbox=why, probes=rows, backends_detected=avail,
                       vm_grade_isolation="refused: not available on this host (no KVM, no user namespaces)")
    acceptance.caveat("Isolation is an unprivileged uid with Landlock and seccomp inside an unprivileged container, "
                      "not a VM: a host-kernel exploit through a permitted system call would escape it. All sandbox "
                      "workers share one uid; signalling, resource limits and scheduling calls on other processes "
                      "are filtered, the shared uid itself remains.")


# ================================================================================================ B
def _verified_gain(world, actor: str, endowment: int) -> int:
    """Cash of the participant and of every organisation it owns, minus what it started with: read from the ledger
    by the evaluator, whatever the participant did to earn it."""
    from crucible.goals.metrics import m_cash
    orgs = [m["org"] for m in world.state.find("members", "person", actor) if m["role"] == "owner"]
    return int(m_cash(world, actor) + sum(m_cash(world, o) for o in orgs)) - int(endowment)


def _income_by_kind(world, actor: str) -> dict:
    st = world.state
    owners = {actor} | {m["org"] for m in st.find("members", "person", actor) if m["role"] == "owner"}
    out: dict[str, int] = {}
    for o in sorted(owners):
        for a in st.find_ids("accounts", "owner", o):
            for lid in st.find_ids("acct_txns", "acct", a):
                e = st.tables["acct_txns"][lid]
                kind = st.tables["txns"][e["txn"]]["kind"]
                if e["amount"] > 0 and kind not in ("endowment", "transfer"):
                    out[kind] = out.get(kind, 0) + int(e["amount"])
    return out


@pytest.mark.sandbox
def test_B_multiple_viable_strategies(acceptance):
    acceptance.claim("At least two materially different strategies achieve the same independently verified "
                     "objective in multiple world instances, without a hard-coded intended action sequence.")
    from crucible.agents.scripted import IdlePolicy, call, ok
    from crucible.research.llm_eval import RigExperimenter
    from crucible.usefulness.agents import ExampleInductor

    class Operator(RigExperimenter):
        """Leases the facility, then finds out by experiment what its levers do."""

        def script(self):
            r = yield call("rig.take")
            if ok(r):
                yield from super().script()

    days, endowment, target = 20, 2_000_000, 20_000      # the same objective for everybody: +$200 in 20 days
    seeds = (1, 2, 3, 4)
    rows, hidden = {}, {}
    for seed in seeds:
        for name, make in (("sell_software", ExampleInductor), ("operate_facility", lambda: Operator(horizon=24 * days)),
                           ("idle", IdlePolicy)):
            if name == "idle" and seed > 2:
                continue
            w = World.create(generate({"family": "ow_works"}, seed), world_id=f"acc-b-{seed}-{name}")
            w.system_call("admin.join", {"handle": "ann", "endowment": endowment})
            pol = make()
            Runner(w, [("p:ann", pol)], dt=1800, seed=seed, record=True).run(48 * days)
            tools = sorted({e.payload["tool"] for e in w.log if e.type == "cmd" and e.actor == "p:ann"})
            gain = _verified_gain(w, "p:ann", endowment)
            rows.setdefault(name, {})[seed] = {"gain_cents": gain, "objective_met": gain >= target,
                                               "income_by_kind": _income_by_kind(w, "p:ann"), "tools": tools}
            assert w.check_invariants() == [] and not [e for e in w.log if e.type == "proc.failed"], (seed, name)
            hidden[seed] = {m["name"]: m["config"] for m in w.spec.modules if m["name"] in ("use", "rig")}
            w.close()
    assert len({json.dumps(v, sort_keys=True, default=str) for v in hidden.values()}) == len(seeds)  # different worlds
    met = {n: sorted(s for s, r in x.items() if r["objective_met"]) for n, x in rows.items()}
    assert met["idle"] == []
    both = sorted(set(met["sell_software"]) & set(met["operate_facility"]))
    assert len(both) >= 2, met
    for seed in both:          # materially different: different services used, different sources of the money
        a, b = rows["sell_software"][seed], rows["operate_facility"][seed]
        assert "cloud.deploy" in a["tools"] and "market.list" in a["tools"] and not any(t.startswith("rig.") for t in a["tools"])
        assert "rig.set" in b["tools"] and "cloud.deploy" not in b["tools"] and "market.list" not in b["tools"]
        assert a["income_by_kind"].get("sale", 0) > 0 and "rig_settle" not in a["income_by_kind"]
        assert b["income_by_kind"].get("rig_settle", 0) > 0 and "sale" not in b["income_by_kind"]
    acceptance.measure(objective={"gain_cents": target, "days": days, "endowment_cents": endowment},
                       worlds_where_objective_met=met, worlds_where_both_strategies_met_it=both,
                       results={n: {str(s): {k: v for k, v in r.items() if k != "tools"} for s, r in x.items()}
                                for n, x in rows.items()})
    acceptance.caveat("Both strategies are carried out by scripted competent controls that decide from what they "
                      "observe. This shows the world admits more than one route to the same verified outcome; it "
                      "does not show that a learner finds either.")
    acceptance.caveat("The horizon was changed after a first attempt. With 10 days the software strategy met the "
                      "objective in only 1 of 3 worlds, because a company pays its set-up costs first (it was at "
                      "+$143 and +$286 on day 10 in two worlds). The horizon was then set to 20 days and a fourth, "
                      "previously unseen world was added. The software strategy is not viable in every world: in "
                      "world 3 it was still losing money on day 20.")


# ================================================================================================ D
def test_D_structural_transfer(tmp_path, acceptance):
    acceptance.claim("A learner trained on one family of rules meets a different mechanism family; its performance "
                     "and adaptation are measured against learning from scratch.")
    from crucible.research import protocols as P
    from crucible.research import tasks as T
    from crucible.research.learner import EnvSpec
    spec = EnvSpec()
    source = ["scm_additive", "scm_gate"]
    assert T.MECH_FAMILIES["scm_invu"].transfer == "structural" and "scm_invu" not in source
    targets = {"structural": {"category": "structural", "families": ["scm_invu"],
                              "eval": T.eval_tasks(["scm_invu"], 6),
                              "train": P.task_stream(["scm_invu"], 500_000)}}
    r = P.transfer(P.from_registry("model_based", spec), source, targets, source_budget=6000, adapt_budget=3000,
                   eval_every=1500, seeds=(0,), spec=spec, n_envs=8, workdir=str(tmp_path), label="acceptance-D")
    o = r["targets"]["structural"]
    assert r["source"]["families"] == sorted(source) and o["families"] == ["scm_invu"]
    assert not set(o["families"]) & set(r["source"]["families"])
    a = o["adaptation"]
    assert a["same_training_tasks"] is True and a["budget_env_steps"] == 3000
    costs = a["cost"]
    assert costs["pretrained"][0]["env_steps"] == costs["scratch"][0]["env_steps"]       # equal experience
    assert o["zero_shot"]["gain"]["n"] == 6 and a["auc"]["diff"]["n"] == 6               # paired on the same tasks
    assert isinstance(o["negative_transfer"], bool)
    recorded = {n: _evidence(n) for n in ("ow_benchmark_transfer_default.json",) if _evidence(n)}
    acceptance.measure(learner="model_based", source_families=source, target_family="scm_invu",
                       source_steps=6000, adaptation_steps=3000, zero_shot_gain_over_untrained=o["zero_shot"]["gain"],
                       adaptation_auc={"pretrained": a["auc"]["pretrained"], "scratch": a["auc"]["scratch"],
                                       "paired_difference": a["auc"]["diff"]},
                       final_score={"pretrained": a["final"]["pretrained"], "scratch": a["final"]["scratch"]},
                       negative_transfer=o["negative_transfer"], recorded_longer_runs=sorted(recorded))
    acceptance.caveat("This test establishes that the protocol measures structural transfer on equal experience; it "
                      "asserts no particular result. With this small budget and n=6 the measured difference is "
                      "reported as it came out.")


# ================================================================================================ E
def test_E_autonomous_competition(acceptance):
    acceptance.claim("Two independently controlled agents compete; their policies adapt in response to each other's "
                     "actions; the environment's changes result from real executable decisions.")
    from crucible.research.learner import EnvSpec
    from crucible.selfplay import modes as M
    from crucible.selfplay import profiles as PF
    from crucible.selfplay.arena import Individual, KernelSelfPlayEnv, SelfPlayEnv, arena_task, play
    from crucible.selfplay.learners import make_learner
    spec = EnvSpec()
    t = arena_task("abm_duopoly", 0)
    trained, net = {}, {}
    for opp in ("aggressive", "passive"):
        env = SelfPlayEnv([t] * 4, spec)
        me = Individual(f"vs_{opp}", make_learner(spec, "spsa", 5), "own", seat=0)
        other = PF.profile_individual(t, 1, opp, spec)
        for it in range(24):
            play(env, [me, other], noise=2000 + 4 * it)
        trained[opp] = me
    for tr, me in trained.items():
        for opp in ("aggressive", "passive"):
            net[f"trained_vs_{tr}|plays_{opp}"] = M.evaluate(t, [me, PF.profile_individual(t, 1, opp, spec)],
                                                             n_slots=6)["net_per_tick"][0]
    dist = float(np.linalg.norm(np.asarray(trained["aggressive"].learner.theta) -
                                np.asarray(trained["passive"].learner.theta)))
    own = net["trained_vs_aggressive|plays_aggressive"] + net["trained_vs_passive|plays_passive"]
    cross = net["trained_vs_passive|plays_aggressive"] + net["trained_vs_aggressive|plays_passive"]
    assert dist > 0.05 and own > cross, (dist, net)      # what it learned depends on whom it learned against
    # two independent learners in one market, each moving the other's payoff, as recorded tool calls in a world
    env = SelfPlayEnv([t] * 4, spec)
    pair = [Individual(f"p{s}", make_learner(spec, "spsa", 11 + s), "own", seat=s) for s in range(2)]
    play(env, pair, noise=3000)                      # a learner's parameters exist from its first episode
    start = [np.array(i.learner.theta, copy=True) for i in pair]
    for it in range(1, 12):
        play(env, pair, noise=3000 + 4 * it)
    assert all(np.linalg.norm(np.asarray(i.learner.theta) - s) > 0.01 for i, s in zip(pair, start))
    k = KernelSelfPlayEnv(t, spec, handles=("ana", "ben"))
    obs = k.reset()
    for i in pair:
        i.learner.freeze(True)
        i.learner.reset_context(1)
    for _ in range(24):
        obs, _, _, _ = k.step([i.learner.act(o, explore=False) for i, o in zip(pair, obs)])
    assert all(sum(1 for c in k.calls[s] if c["tool"] == "rig.set") == 24 for s in range(2))
    assert World.replay(k.world.spec, [e.to_dict() for e in k.world.log]).digest() == k.world.digest()
    recorded = _evidence("ow_selfplay_A.json")
    acceptance.measure(two_by_two_net_per_tick=net, own_opponent_total=own, other_opponent_total=cross,
                       policy_distance_between_opponents=dist, kernel_ticks=24,
                       recorded_experiment_present=recorded is not None)
    acceptance.caveat("The learners are stochastic-approximation controllers on a few policy parameters; opponents "
                      "in the 2x2 are scripted profiles. Richer adaptation in joint training is reported from the "
                      "recorded self-play experiment, with its uncertainty.")


# ================================================================================================ F
def test_F_cooperation(acceptance):
    acceptance.claim("Independent agents achieve a difficult objective by coordinating complementary capabilities; "
                     "removing cooperation changes the outcome under controlled conditions.")
    from crucible.research.learner import EnvSpec
    from crucible.research.stats import paired_diff
    from crucible.selfplay import modes as M
    from crucible.selfplay.arena import Individual, SelfPlayEnv, arena_task, play
    from crucible.selfplay.learners import make_learner
    spec = EnvSpec()
    rows = []
    for k in (0, 1, 2):
        t = arena_task("tech_handoff", k)
        env = SelfPlayEnv([t] * 6, spec)
        inds = [Individual(f"s{s}", make_learner(spec, "spsa", 10 + s), "joint", seat=s) for s in range(2)]
        for it in range(16):
            play(env, inds, noise=1000 + 6 * it)
        pair = M.evaluate(t, inds, n_slots=6)["joint_net_per_tick"]      # identical worlds and noise in every arm
        base = M.evaluate(t, [None, None], n_slots=6)["joint_net_per_tick"]
        solo0 = M.evaluate(t, [inds[0], None], n_slots=6)["joint_net_per_tick"]
        solo1 = M.evaluate(t, [None, inds[1]], n_slots=6)["joint_net_per_tick"]
        rows.append({"world": k, "trained_pair": pair, "both_default": base, "only_upstream_trained": solo0,
                     "only_downstream_trained": solo1})
    p = [r["trained_pair"] for r in rows]
    best_solo = [max(r["only_upstream_trained"], r["only_downstream_trained"]) for r in rows]
    assert all(a > b for a, b in zip(p, best_solo)), rows
    assert all(r["trained_pair"] > r["both_default"] for r in rows), rows
    acceptance.measure(family="tech_handoff", joint_net_per_tick=rows, pair_minus_best_single=paired_diff(p, best_solo))
    acceptance.caveat("Three world instances; a paired interval over three is reported but flagged low-n. "
                      "'Cooperation' here is two independent learners sharing a team objective with no communication "
                      "channel; each sees only its own gauges.")


# ================================================================================================ K
def test_K_architecture_neutral_evaluation(acceptance):
    acceptance.claim("Different RL architectures use the environment through consistent interfaces, with equivalent "
                     "permissions and evaluation budgets.")
    from test_learners import easy_task
    from crucible.research import learners as L
    from crucible.research import protocols as P
    from crucible.research.bridge import KernelMechEnv
    from crucible.research.envs import MechVecEnv
    from crucible.research.learner import EnvSpec, Learner
    from crucible.research import tasks as T
    spec = EnvSpec()
    names = L.available()
    kinds = {n: L.describe(n)["kind"] for n in names}
    assert {"model_free", "model_based"} <= set(kinds.values()) and len(set(kinds.values())) >= 5
    tasks = [easy_task(k) for k in range(4)]
    consumed = {}
    for n in names:
        lr = L.make(n, spec, seed=0)
        assert isinstance(lr, Learner)
        env = MechVecEnv(tasks, spec)
        obs = env.reset()
        lr.reset_context(len(tasks))
        while not env.done.all():
            a = lr.act(obs, explore=False)
            assert a.shape == (len(tasks), spec.act_dim) and np.all(np.abs(a) <= 1.0 + 1e-6), n
            obs, _, _, _ = env.step(a)
        consumed[n] = env.steps
    assert len(set(consumed.values())) == 1                 # every architecture: the same interaction budget
    # the same learner through ordinary tool calls in a kernel world gets the same observations and rewards
    real = T.build_tasks([("scm_additive", T.VAL_SEED0)])[0]
    for n in ("ppo_gru", "model_based"):
        la, lb = L.make(n, spec, seed=3), L.make(n, spec, seed=3)
        sim, ker = MechVecEnv([real], spec), KernelMechEnv(real, spec)
        oa, ob = sim.reset(), ker.reset()
        la.reset_context(1)
        lb.reset_context(1)
        for _ in range(12):
            a, b = la.act(oa, explore=False), lb.act(ob, explore=False)
            assert np.array_equal(a, b), n
            oa, ra, _, _ = sim.step(a)
            ob, rb, _, _ = ker.step(b)
            assert np.array_equal(oa, ob) and ra[0] == rb[0], n
    # a comparison under one budget: identical tasks, schedule and experience for each architecture
    r = P.compare({"ppo_gru": P.from_registry("ppo_gru", spec, hidden=32, rollout_steps=480),
                   "model_based": P.from_registry("model_based", spec)},
                  [easy_task(k) for k in range(4)], {"fresh": [easy_task(k) for k in range(50, 54)]}, 1920, 960,
                  seeds=(0,), spec=spec, n_envs=8, keep_runs=True, label="acceptance-K")
    pts = {n: [(p["env_steps"]) for p in x[0]["points"]] for n, x in r["runs"].items()}
    assert pts["ppo_gru"] == pts["model_based"], pts
    ids = {n: x[0]["final"]["fresh"]["tasks"] for n, x in r["runs"].items()}
    assert ids["ppo_gru"] == ids["model_based"]
    acceptance.measure(architectures=kinds, env_steps_each=consumed, comparison_schedule_env_steps=pts["ppo_gru"],
                       same_evaluation_tasks=True, kernel_equals_simulator_for=["ppo_gru", "model_based"])
    acceptance.caveat("Language-model agents use the same world through the text tool interface (agents/llm.py) "
                      "rather than this numeric contract; both reach the facility through the same `rig.*` tools.")


# ================================================================================================ L
def test_L_parametric_learning(acceptance):
    acceptance.claim("A neural agent improves after training; updated weights outperform the original checkpoint on "
                     "fresh tasks with episodic context reset; the environment supports this even if a learner "
                     "fails.")
    from test_learners import easy_task
    from crucible.research import protocols as P
    from crucible.research.learner import EnvSpec
    spec = EnvSpec()
    train, fresh = [easy_task(k) for k in range(8)], [easy_task(k) for k in range(100, 112)]
    r = P.internalisation(P.from_registry("ppo_gru", spec, hidden=32, rollout_steps=960, lr=1e-3), train, fresh,
                          28_800, seed=0, spec=spec, n_envs=16, label="acceptance-L")
    assert r["valid"] is True and r["violations"] == []
    assert r["frozen_unchanged"]["original"] is True and r["frozen_unchanged"]["trained"] is True
    assert r["update_count"] > 0 and r["env_interactions"] >= 28_800
    assert r["verdict"] == "internalised" and r["gain"]["lo"] > 0, r["gain"]
    # the protocol does not manufacture a gain: a learner with no weights gets none on the same tasks
    null = P.internalisation(P.from_registry("probe_exploit", spec), train[:2], fresh[:6], 480, seed=0, spec=spec,
                             n_envs=8, label="acceptance-L-null")
    assert null["valid"] is True and null["verdict"] == "no_gain_detected"
    acceptance.measure(learner="ppo_gru (recurrent PPO)", weight_updates=r["update_count"],
                       env_interactions=r["env_interactions"], compute=r["compute"], gain_on_fresh_tasks=r["gain"],
                       original=r["original"]["score"], trained=r["trained"]["score"],
                       verdict=r["verdict"], control_without_weights={"verdict": null["verdict"], "gain": null["gain"]})
    acceptance.caveat("Shown on easy hand-made mechanisms from one structural family, where learning is cheap. On the "
                      "real benchmark families trained PPO improved far less; see the learner table.")


# ================================================================================================ N
@pytest.mark.slow
def test_N_clean_installation(tmp_path, acceptance):
    acceptance.claim("A fresh checkout can install and run the supported environment without relying on "
                     "undocumented local state.")
    import shutil
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not installed")
    repo = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    co = tmp_path / "checkout"
    subprocess.run(["git", "clone", "-q", "--no-hardlinks", repo, str(co)], check=True)
    subprocess.run(["git", "checkout", "-q", head], cwd=co, check=True)
    home = tmp_path / "home"
    home.mkdir()
    env = {"PATH": os.environ["PATH"], "HOME": str(home), "UV_PYTHON_PREFERENCE": "only-system",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    vpy = str(tmp_path / "venv" / "bin" / "python")
    subprocess.run([uv, "venv", "-q", "--python", sys.executable, str(tmp_path / "venv")], env=env, check=True, timeout=300)
    inst = subprocess.run([uv, "pip", "install", "-q", "--python", vpy, f"{co}/crucible[dev]"], env=env,
                          capture_output=True, text=True, timeout=900)
    assert inst.returncode == 0, inst.stderr[-2000:]
    script = ("import crucible, json; crucible.load_all_modules()\n"
              "from crucible.worldgen import generate, families\n"
              "from crucible.core.kernel import World\n"
              "out = {'families': len(families())}\n"
              "for fam in ('mech_scm_additive', 'mech_abm_utility', 'mech_tech_pipeline', 'ow_tiny'):\n"
              "    w = World.create(generate({'family': fam}, 1)); w.advance_time(12 * 3600)\n"
              "    r = World.replay(w.spec, [e.to_dict() for e in w.log])\n"
              "    assert not w.check_invariants() and r.digest() == w.digest(), fam\n"
              "    out[fam] = w.seq\n"
              "print(json.dumps(out))\n")
    run = subprocess.run([vpy, "-c", script], cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stderr[-2000:]
    smoke = json.loads(run.stdout.strip().splitlines()[-1])
    tests = subprocess.run([vpy, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_kernel.py",
                            "tests/test_research_stats.py", "tests/test_mech_scm.py"], cwd=f"{co}/crucible", env=env,
                           capture_output=True, text=True, timeout=1500)
    assert tests.returncode == 0, tests.stdout[-2000:]
    acceptance.measure(commit=head, install="uv pip install crucible[dev] into an empty venv, empty HOME",
                       smoke_worlds=smoke, tests_in_clean_checkout=tests.stdout.strip().splitlines()[-1])
    acceptance.caveat("Installs the core and dev extras. The web, browser, PostgreSQL and RL extras, the Crucible "
                      "sandbox's root requirement and model weights are documented separately and are not part of "
                      "this check.")


# ================================================================================================ O
def test_O_independent_generalization(acceptance):
    acceptance.claim("Performance is evaluated on world families not used for tuning or training.")
    from crucible.research import learners as L
    from crucible.research import protocols as P
    from crucible.research import tasks as T
    from crucible.research.learner import EnvSpec
    spec = EnvSpec()
    train = set(T.families_by("train", seats=None))
    held = ["scm_feedback", "scm_regime", "abm_network_fx", "tech_contention"]
    assert not set(held) & train and all(T.MECH_FAMILIES[f].split == "test" for f in held)
    assert {T.MECH_FAMILIES[f].transfer for f in held} >= {"structural", "cross_impl"}
    tasks = T.build_tasks([(f, T.VAL_SEED0 + i) for f in held for i in range(2)], workers=1)
    subjects = {"hold": L.make("hold", spec), "probe_exploit": L.make("probe_exploit", spec)}
    ck_dir = os.path.join(EXPERIMENTS, "checkpoints")
    curves = {}
    for name in ("ppo_gru", "model_based"):
        run = _evidence(f"ow_learners_curve_{name}_s0.json")
        ck = os.path.join(ck_dir, f"ow_learners_{name}_s0.pt")
        if run is None or not os.path.exists(ck):
            continue
        assert set(run["families"]) <= train              # the checkpoint never saw a held-out family
        lr = L.make(name, spec, seed=0, **(run.get("kwargs") or {}))
        lr.load(ck)
        subjects[f"{name}_trained"] = lr
        curves[name] = run["families"]
    assert len(subjects) >= 3
    out = {}
    for n, lr in subjects.items():
        ev = P.evaluate(lr, tasks, spec=spec, n_envs=8, label=f"acceptance-O-{n}")
        assert ev["frozen"]["unchanged"] is True, n
        out[n] = {"score": ev["aggregate"]["score"], "by_transfer": {k: v["score"] for k, v in ev["by_transfer"].items()},
                  "success": ev["aggregate"]["success"]}
    assert all(o["score"]["n"] == len(tasks) for o in out.values())
    acceptance.measure(evaluated_families=held, training_families=sorted(train), results=out,
                       checkpoints_trained_on=curves,
                       sealed_result_present=_evidence("ow_benchmark_sealed_default.json") is not None)
    acceptance.caveat("These are test-split families at public validation seeds, so the procedure, not the numbers, "
                      "is what this test establishes. The final figures come from one opening of the sealed set.")
