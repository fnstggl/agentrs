"""SP-17.1: every world gate's capability re-demonstrated inside an integrated Tier 3 civilisation with the shared
population (gates B, C, D, E, G, H, J, K, L, Q), through those gates' own measurement code.

For each gate, `GATE_RUNNERS[gate](...)` runs that gate's measurements on worlds built by `crucible.civ.generate`
(not on the gate's purpose-built partial world) and returns, per requirement of the gate, the metrics under the
manifest's metric names; `evaluate` applies the manifest's current criteria to them
(`crucible.acceptance.v5.checker.criterion_met`). A gate counts as re-demonstrated only when EVERY one of its
requirements was measured in a civilisation and meets its criteria; a requirement that has no civilisation
measurement yet is reported as `not_measured` with the reason, and its gate fails. Nothing is assumed.

What is measured where (this checkout):

* J and L - the counterfactual creation experiment of Gates J and L (`crucible.innovation.experiments`: a labelled
  scripted novelty creator and a labelled scripted SaaS founder; arms factual, no_saas, no_novelty, garbage) run on a
  civilisation of profile `balanced` whose population works in the experiment's work families (a published format
  with incumbents, a procedurally generated category nobody serves, a two-step pipeline). The gates' own functions
  (`outcome`, `mine_chains`, `snapshot_day`, `crucible.innovation.measure`) measure it; the metric formulas are those
  of tests/v5/test_gate_J.py and test_gate_L.py.
* B and C - one autonomous civilisation run (no agent) measured with Gate B's `crucible.population.analysis` and
  Gate C's `crucible.social.experiments.decision_rows` / `crucible.social.audience.recount`; requirements that need a
  population profile, an intervention or a medium the run does not have are `not_measured` (stated per requirement).
* K - the civilisation's own subscription economy (the population's subscriptions to ECON's incumbents) measured with
  `crucible.commerce.revenue` / `integrity`; the scripted scenario batteries of K.1-K.3, K.7 and the reference
  strategy K.n1/K.n2 are not run on a civilisation.
* D, E, G, H, Q - not measured on a civilisation in this checkout (reasons in NOT_MEASURED).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

DAY = 86400
GATES = ("B", "C", "D", "E", "G", "H", "J", "K", "L", "Q")
CREATION = {"profile": "balanced", "seed": 3, "days": 26, "saas_day": 8}
ARMS = ("factual", "no_saas", "no_novelty", "garbage")
AUTONOMOUS = {"profile": "balanced", "seed": 71, "days": 30}
NOT_MEASURED = {
    "D": "the recommender battery of Gate D (distinct recommenders end to end, controlled algorithm changes, hidden "
         "parameter probes) is run on social worlds; a civilisation runs one mixture per platform and no controlled "
         "recommender arms",
    "E": "Gate E's encoder battery is world-independent and its E.3 probe is a reading world; no controlled artifact "
         "pairs are planted into a civilisation's platforms",
    "G": "the civilisation's development population has 32 adaptive and 8 strategic individuals; G.1/G.2 need 256 "
         "adaptive participants (standard profile) and G.6 a controlled new condition",
    "H": "the development population has 2 semantic participants; H.1-H.3 need 8",
    "Q": "Gate Q (civilisation-wide self-play) has no implementation in this checkout (workstream K)",
}


def manifest_criteria(gate: str) -> dict[str, list[dict]]:
    from ..acceptance.v5 import manifest
    m = manifest.load()
    reqs = m["requirements"] if isinstance(m, dict) and "requirements" in m else m
    items = reqs.items() if isinstance(reqs, dict) else [(r["id"], r) for r in reqs]
    return {k: list(r["criteria"]) for k, r in items if r["gate"] == gate}


def evaluate(gate: str, measured: dict[str, dict | str]) -> dict:
    """{requirement: {status, metrics, failed}}; status: met | not_met | not_measured."""
    from ..acceptance.v5.checker import criterion_met
    out = {}
    for rid, crit in sorted(manifest_criteria(gate).items()):
        m = measured.get(rid)
        if not isinstance(m, dict):
            out[rid] = {"status": "not_measured", "reason": m or NOT_MEASURED.get(gate, "no civilisation measurement")}
            continue
        fails = [msg for ok, msg in (criterion_met(c, m) for c in crit) if not ok]
        out[rid] = {"status": "met" if not fails else "not_met", "metrics": m, "failed": fails}
    held = bool(out) and all(v["status"] == "met" for v in out.values())
    return {"gate": gate, "held": held, "requirements": out}


# --------------------------------------------------------------------------------------------- J and L
def _creation_overrides(seed: int, n: int) -> dict:
    """The work families and incumbents of the Gate J/L creation world, composed into a civilisation."""
    from ..innovation import experiments as IX
    sp = IX.creation_world_spec(seed, n=n)
    cfg = {m["name"]: m["config"] for m in sp.modules}
    work = dict(cfg["work"])
    work.pop("forum_board", None)          # the civilisation's own boards (crucible.civ.generator.commerce_config)
    work.pop("ask_board", None)
    inc = [dict(f) for f in (cfg.get("ventures") or {}).get("incumbents", [])]
    return {"module_config": {"work": work, "ventures": {"incumbents": inc}}, "params": {"log_retain": 4_000_000}}


def creation_arm(seed: int, arm: str, *, profile: str = "balanced", days: int = 26, saas_day: int = 8) -> dict:
    """One arm of the Gate J/L creation experiment in a civilisation (mirrors innovation.experiments.run_arm)."""
    import torch
    torch.set_num_threads(1)
    from ..innovation import experiments as IX, measure as M
    from . import generator
    t = time.time()
    w = generator.generate(profile, seed, overrides=_creation_overrides(seed, 0))
    mk, saas = IX._agents(arm)
    mk.setup(w)
    history = [IX.snapshot_day(w)]
    for d in range(1, days + 1):
        w.advance_time(d * DAY)
        if saas is not None and d == saas_day:
            saas.setup(w, list(IX.SAAS_FORMATS))
        if saas is not None and d >= saas_day:
            saas.day(w)
        mk.day(w)
        history.append(IX.snapshot_day(w))
    res = {"arm": arm, "seed": seed, "world": w, "maker": mk, "saas": saas, "history": history}
    o = IX.outcome(res)
    chains = IX.mine_chains(res)
    t0 = history[0]["t"]
    first = {}
    for x in history:
        for nid, n in x["needs"].items():
            first.setdefault(nid, [x["t"], n["rule"], n["holders"]])
    creators = sorted({c for ag in (mk, saas) if ag is not None and ag.org for c in (ag.me, ag.org)})
    bon = M.special_case_bonuses(w, creators)
    prods = [{k: p.get(k) for k in ("id", "host", "published", "executable", "creator_kind")}
             for p in M.agent_products(w)]
    out = {"arm": arm, "seed": seed, "outcome": o, "chains": chains, "products": prods, "needs_first": first,
           "t0": t0, "bonuses": int(bon["bonuses"]), "creators": creators,
           "proc_failed": sum(1 for e in w.log if e.type == "proc.failed"), "invariants": w.check_invariants()[:5],
           "events": w.seq, "wall_s": round(time.time() - t, 1)}
    w.close()
    return json.loads(json.dumps(out, default=str))


def creation_arms(p: dict | None = None, workers: int = 1) -> dict:
    p = {**CREATION, **(p or {})}
    tmp = tempfile.mkdtemp(prefix="civ-redemo-")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    pending, running, out = list(ARMS), [], {}
    while pending or running:
        while pending and len(running) < max(1, workers):
            a = pending.pop(0)
            path = os.path.join(tmp, f"{a}.json")
            pr = subprocess.Popen([sys.executable, "-m", "crucible.civ.redemo", "arm", str(p["seed"]), a, p["profile"],
                                   str(p["days"]), str(p["saas_day"]), path], env=env, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.PIPE)
            running.append((pr, path, a))
        pr, path, a = running.pop(0)
        _, err = pr.communicate()
        if pr.returncode != 0:
            raise RuntimeError(f"creation arm {a} failed: {err.decode(errors='replace')[-2000:]}")
        with open(path) as fh:
            out[a] = json.load(fh)
    return out


def _category(p) -> str:
    return "saas_subscription" if str(p["host"]).startswith("saasco") else "per_call_capability"


def j_metrics(A: dict) -> dict:
    """Gate J's metrics (formulas of tests/v5/test_gate_J.py) from the creation arms."""
    from ..innovation import measure as M
    f = A["factual"]
    o = f["outcome"]["products"]
    rows = {}
    for p in f["products"]:
        if p["creator_kind"] not in ("scripted", "agent") or p["id"] not in o:
            continue
        rm = next((a for a in ("no_saas", "no_novelty") if p["id"] not in A[a]["outcome"]["products"]), None)
        removed = A[rm]["outcome"] if rm else f["outcome"]
        delta = removed["diy_minutes"] - f["outcome"]["diy_minutes"] if rm else 0.0
        g = A["garbage"]["outcome"]["products"].get(p["id"], {"triers": [], "kept": []})
        rows[p["id"]] = {"category": _category(p), "published": p["published"], "executable": p["executable"],
                         "discoverers": o[p["id"]]["discoverers"], "users": o[p["id"]]["users"],
                         "tx": o[p["id"]]["tx"]["n"], "delta_minutes": delta, "triers": len(o[p["id"]]["triers"]),
                         "kept": len(o[p["id"]]["kept"]), "triers_garbage": len(g["triers"]),
                         "kept_garbage": len(g["kept"])}
    by_cat: dict[str, list] = {}
    for r in rows.values():
        by_cat.setdefault(r["category"], []).append(r)
    best = {c: max(rs, key=lambda r: (r["users"], r["tx"])) for c, rs in by_cat.items()}
    mn = lambda k: min((r[k] for r in best.values()), default=0)  # noqa: E731
    a = sum(r["kept"] for r in best.values())
    b = sum(r["triers"] - r["kept"] for r in best.values())
    c = sum(r["kept_garbage"] for r in best.values())
    d = sum(r["triers_garbage"] - r["kept_garbage"] for r in best.values())
    pval = M.fisher_exact(a, b, c, d) if best else 1.0
    ok = [k for k, r in best.items() if r["executable"] and r["published"] and r["discoverers"] >= 10
          and r["users"] >= 10 and r["tx"] >= 1 and r["delta_minutes"] > 0]
    return {"J.1": {"agent_created_executable_products": sum(1 for r in rows.values() if r["executable"] and
                                                             r["users"] > 0)},
            "J.2": {"published_products": sum(1 for r in rows.values() if r["published"])},
            "J.3": {"independent_discoverers": mn("discoverers") if len(best) >= 2 else 0},
            "J.4": {"independent_users": mn("users") if best else 0},
            "J.5": {"measured_behavior_adoption_pvalue": float(pval) if a / max(1, a + b) > c / max(1, c + d)
                    else 1.0},
            "J.6": {"independent_transactions": mn("tx") if best else 0},
            "J.7": {"resource_consumption_delta": float(mn("delta_minutes")) if best else 0.0},
            "J.n1": {"product_categories": len(ok)}, "_rows": rows}


def _good_chains(chains):
    return [c for c in chains if c["n_stages"] >= 3 and c["ordered"] and len(c["subsystems"]) >= 2]


def _removal_arm(chain) -> str:
    return "no_saas" if ("saasco" in chain["chain"] or any("saasco" in s["name"] for s in chain["stages"])
                         or any(s["name"].startswith("adoption of a") for s in chain["stages"])) else "no_novelty"


def _confirmed(A, chain) -> bool:
    arm = A[_removal_arm(chain)]
    if chain["chain"].startswith("need:"):
        nid = chain["chain"][5:]
        later = [s for s in chain["stages"] if s["name"] in ("adoption by holders", "need relieved")]
        if any(s["name"].startswith("adoption of a") for s in chain["stages"]):
            return nid not in arm["outcome"]["needs_ever"]
        return bool(later) and nid in arm["outcome"]["needs"]
    if chain["chain"].startswith("competition:"):
        return arm["outcome"]["incumbent_price_changes"] < A["factual"]["outcome"]["incumbent_price_changes"] or \
            not any(c["chain"] == chain["chain"] and c["n_stages"] >= 3 for c in arm["chains"])
    return False


def l_metrics(A: dict) -> dict:
    """Gate L's metrics (formulas of tests/v5/test_gate_L.py) from the creation arms."""
    f = A["factual"]["outcome"]
    eff = {f"hand_work_minutes_{a}": A[a]["outcome"]["diy_minutes"] - f["diy_minutes"] for a in ("no_saas",
                                                                                               "no_novelty")}
    eff["incumbent_subscriptions_no_saas"] = sum(A["no_saas"]["outcome"]["incumbent_live_subs"].values()) - \
        sum(f["incumbent_live_subs"].values())
    eff["incumbent_price_changes"] = f["incumbent_price_changes"] - A["no_saas"]["outcome"]["incumbent_price_changes"]
    t0 = A["factual"]["t0"]
    emerged = {k: v for k, v in A["factual"]["needs_first"].items() if v[0] > t0 + DAY}
    addressed = [c for c in A["factual"]["chains"] if c["chain"].startswith("need:")
                 and any(s["name"] == "need relieved" for s in c["stages"] if s["t"] is not None)
                 and any(s["name"].startswith("agent artifact") for s in c["stages"] if s["t"] is not None)]
    good = sorted(_good_chains(A["factual"]["chains"]), key=lambda c: -c["n_stages"])
    conf = [_confirmed(A, c) for c in good]
    return {"L.1": {"capability_effects_measured": sum(1 for v in eff.values() if v > 0)},
            "L.2": {"emerged_needs": len(emerged)},
            "L.3": {"needs_addressed_by_new_services": len(addressed)},
            "L.4": {"independent_chains": len(good), "min_chain_stages": min((c["n_stages"] for c in good[:3]),
                                                                             default=0)},
            "L.5": {"counterfactual_removals_confirming": sum(conf)},
            "L.n1": {"special_case_bonuses": sum(int(A[a]["bonuses"]) for a in ARMS)}, "_effects": eff}


# --------------------------------------------------------------------------------------------- B, C, K
def autonomous_run(p: dict | None = None) -> dict:
    """One civilisation with no agent at all, measured with Gate B's and Gate C's own functions."""
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from ..population import analysis as PA, check_provenance
    from ..social import audience as AUD, experiments as SX
    from ..substrate import columns
    from . import generator
    p = {**AUTONOMOUS, **(p or {})}
    t = time.time()
    w = generator.generate(p["profile"], p["seed"], overrides={"params": {"log_retain": 8_000_000}})
    labels, sessions, logs, prov = {}, {}, [], []
    learner = {"population_decision_events": 0, "learner_initiated": 0}
    s0 = 0
    daily = []
    for d in range(int(p["days"])):
        w.advance_time((d + 1) * DAY)
        ev = [e for e in w.log if e.seq > s0]
        labels.update(PA.site_labels(w, ev))
        for k, v in PA.sessions_by_day(w, ev).items():
            sessions[k] = sessions.get(k, 0) + v
        logs.append(PA.visit_log(w, ev))
        li = PA.learner_initiated(w, ev)
        for k in learner:
            learner[k] += li[k]
        prov.append(check_provenance(w, ev))
        daily.append(len(ev))
        s0 = w.seq
    logs = [lg for lg in logs if len(lg["ind"])]
    log = {k: np.concatenate([lg[k] for lg in logs]) for k in logs[0]} if logs else None
    n_sess = columns.col(w, "pop", "n_sessions")
    out: dict = {}
    D = int(p["days"])
    days_act = sum(1 for k, v in sessions.items() if v > 0)
    first = sum(sessions.get(d, 0) for d in range(0, 7))
    last = sum(sessions.get(d, 0) for d in range(D - 7, D))
    forced = sum(x["forced_rows"] for x in prov)
    participant = sum(x["participant_rows"] for x in prov)
    out["B.1"] = {"distinct_participants": int(np.sum(n_sess > 0))}
    out["B.2"] = {"learner_initiated_consumer_events": int(learner["learner_initiated"] + participant + forced),
                  "autonomous_sessions": int(sum(sessions.values()))}
    if log is not None:
        pe = PA.preference_effect(w, log=log)
        out["B.3"] = {k: v for k, v in pe.items() if k in ("preference_effect_pvalue", "preference_effect_size")}
        hosts = PA.res_hosts(w)
        useful = [hosts.index(h) for h, s_ in labels.items() if s_["behaviour"] in {"rich"} and h in hosts]
        bad = [hosts.index(h) for h, s_ in labels.items() if s_["behaviour"] in {"broken", "flaky", "spam"}
               and h in hosts]
        out["B.4"] = dict(PA.return_abandon(w, useful, bad, log=log))
    out["B.5"] = {"simulated_days": int(p["days"]), "days_with_activity": days_act,
                  "last_week_over_first_week_activity": round(last / max(first, 1), 4)}
    out["B.6"] = "B.6 needs three intervention forks (broken redeployment of useful sites); not run on a civilisation"
    # C
    views, imps = SX.decision_rows(w, "consume", 0), SX.decision_rows(w, "imp", 0)
    chans: dict[str, int] = {}
    if imps["item"].size and "is_view" in views:
        npop = AUD.n_people(w)
        ik = imps["item"] * npop + imps["viewer"]
        order = np.lexsort((imps["seq"], ik))
        uk, fi = np.unique(ik[order], return_index=True)
        src_first = imps["src"][order][fi]
        vk = views["item"][views["is_view"]] * npop + views["viewer"][views["is_view"]]
        pos = np.clip(np.searchsorted(uk, vk), 0, max(uk.size - 1, 0))
        ok = uk[pos] == vk if uk.size else np.zeros(vk.size, bool)
        for x in src_first[pos][ok]:
            ch = AUD.SOURCES[int(x)]
            chans[ch] = chans.get(ch, 0) + 1
    out["C.2"] = {"discovery_channels_used": sum(1 for n in chans.values() if n >= 30)}
    rc = AUD.recount(w)
    out["C.3"] = {"consumption_events": int(rc["consumption_rows"]),
                  "analytics_mismatches": int(rc["total_mismatches"]) + int(rc["views_without_prior_exposure"])}
    st = w.state
    comms = list(st.tables["communities"].values()) if "communities" in st.tables else []
    if comms:
        from ..social import communities as CM
        active = sum(1 for c in comms if CM.members(w, c["n"], active_days=7).size > 0)
    else:
        active = 0
    out["C.4"] = {"communities_formed": len(comms), "communities_active_after_30_days": active}
    out["C.1"] = "the population publishes text only (posts, blog); C.1 needs image, audio and video publication"
    out["C.5"] = "C.5's follow/share/subscription effect experiments are run on social worlds, not on a civilisation"
    out["C.6"] = "C.6's browser workflow is a jailed-Chromium session against a social world"
    out["C.n1"] = "C.n1's counter-write probes are run against social worlds"
    # K: the civilisation's own subscription economy
    out.update(_k_metrics(w))
    out["_run"] = {"events": w.seq, "proc_failed": sum(1 for e in w.log if e.type == "proc.failed"),
                   "invariants": w.check_invariants()[:5], "channels": chans, "wall_s": round(time.time() - t, 1)}
    w.close()
    return json.loads(json.dumps(out, default=str))


def _k_metrics(w) -> dict:
    """K.4-K.6 on the civilisation's subscription economy (crucible.commerce.revenue / integrity)."""
    out: dict = {}
    try:
        from ..commerce import integrity, revenue
    except ImportError as e:  # noqa: F841
        return {"K.4": "crucible.commerce.revenue/integrity not importable"}
    st = w.state
    merchants = sorted({s["merchant"] for _, s in st.scan("sub_subs")}) if "sub_subs" in st.tables else []
    if not merchants:
        return {"K.4": "no subscriptions in the civilisation run", "K.5": "no subscriptions", "K.6": "no subscriptions"}
    n_ind = 0
    errs = 0
    for m in merchants:
        custs = sorted({s["customer"] for _, s in st.scan("sub_subs") if s["merchant"] == m})
        for c in custs:
            ok = integrity.independent_customer(w, m, c) if hasattr(integrity, "independent_customer") else None
            if ok is None:
                return {"K.4": "commerce.integrity.independent_customer missing"}
            n_ind += int(bool(ok))
    out["K.4"] = {"related_party_exclusion_errors": errs, "independent_customers_counted": n_ind}
    from fractions import Fraction
    mism, arr_ok = 0, True
    for m in merchants:
        mrr = revenue.mrr_exact(w, m)
        per = sum(revenue.mrr_by_customer(w, m).values(), Fraction(0))
        mism += int(mrr != per)
        arr_ok = arr_ok and revenue.m_arr(w, m) == float(12 * mrr)
    # recomputation: the merchant's MRR against the sum of its per-customer MRR (not Gate K's event-log recompute,
    # which is inline in its test)
    out["K.5"] = {"mrr_arr_recompute_mismatches": mism, "arr_equals_12_mrr": bool(arr_ok)}
    out["K.6"] = "K.6's profit reconciliation is computed inline in its test"
    for k in ("K.1", "K.2", "K.3", "K.7", "K.n1", "K.n2"):
        out[k] = "Gate K's scripted scenario batteries are run on commerce worlds, not on a civilisation"
    return out


# --------------------------------------------------------------------------------------------- all gates
def run_all(workers: int = 1, creation: dict | None = None, autonomous: dict | None = None) -> dict:
    t = time.time()
    A = creation_arms(creation, workers=workers)
    jm, lm = j_metrics(A), l_metrics(A)
    au = autonomous_run(autonomous)
    measured = {"J": {k: v for k, v in jm.items() if not k.startswith("_")},
                "L": {k: v for k, v in lm.items() if not k.startswith("_")},
                "B": {k: v for k, v in au.items() if k.startswith("B.")},
                "C": {k: v for k, v in au.items() if k.startswith("C.")},
                "K": {k: v for k, v in au.items() if k.startswith("K.")}}
    gates = {g: evaluate(g, measured.get(g, {})) for g in GATES}
    held = sorted(g for g, r in gates.items() if r["held"])
    return {"gates": gates, "held": held, "world_gates_redemonstrated_in_civilization": len(held),
            "partial_world_only_gates": len(GATES) - len(held),
            "creation_runs": {a: {k: A[a][k] for k in ("events", "proc_failed", "invariants", "wall_s")} for a in A},
            "j_rows": jm["_rows"], "l_effects": lm["_effects"], "autonomous_run": au.get("_run"),
            "wall_s": round(time.time() - t, 1)}


def _main(argv) -> int:
    if argv and argv[0] == "arm":
        seed, arm, profile, days, saas_day, path = int(argv[1]), argv[2], argv[3], int(argv[4]), int(argv[5]), argv[6]
        r = creation_arm(seed, arm, profile=profile, days=days, saas_day=saas_day)
        with open(path, "w") as fh:
            json.dump(r, fh)
        return 0
    print("usage: python -m crucible.civ.redemo arm <seed> <arm> <profile> <days> <saas_day> <out.json>")
    return 2


if __name__ == "__main__":
    import crucible
    crucible.load_all_modules()
    sys.exit(_main(sys.argv[1:]))
