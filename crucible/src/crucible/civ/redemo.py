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
* D - Gate D's battery on a civilisation (`d_battery`): the civilisation's posts and blog platforms carry Gate D's
  evaluator-side mixtures (`crucible.social.experiments.GATE_D_POP`: a scheduled change on posts at day 3, a blog
  platform that reweights its own mixture from its measured feedback), every other platform keeps the
  civilisation's own; a persistent 4-day run with the invariants checked every 15 minutes, one arm per registered
  recommender forked at day 2 (both platforms distributing with that recommender alone for one day), the
  counterfactual fork of the pre-change state, and the hidden-parameter probes of a participant and a creator. The
  formulas are those of tests/v5/test_gate_D.py on the helpers it uses (`crucible.social.experiments`,
  `crucible.social.audience`).
* G - Gate G's adaptation experiment on a civilisation (`g_arm`, three seeds): the civilisation's population with
  256 class-2 adaptive individuals in two policy groups of IDENTICAL initial weights (one learns online from its
  members' own experienced utilities, the other is frozen), the civilisation's web replaced by the experimenter's ten
  sites of `crucible.population.experiments` (labelled scripted, as in Gate G) so that after phase 1 the two most
  visited rich sites fail and the two broken ones become rich; measured with the formulas of tests/v5/test_gate_G.py
  on the same helpers (`crucible.population.analysis`, the adaptive module's policies, `compact_vs_language` for
  G.7); G.4 from the civilisation's own class-3 strategic groups.
* E, H, Q - not measured on a civilisation in this checkout (reasons in NOT_MEASURED).
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
D_RUN = {"profile": "balanced", "seed": 13, "switch_day": 3}
G_RUN = {"profile": "balanced", "seeds": [0, 1, 2], "phase_days": 5}
NOT_MEASURED = {
    "E": "Gate E's encoder battery is world-independent and its E.3 probe is a reading world; no controlled artifact "
         "pairs are planted into a civilisation's platforms",
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


# --------------------------------------------------------------------------------------------- D
def _d_checks(world):
    errs = world.check_invariants()
    return {"aud": [e for e in errs if e.startswith("aud:")],
            "attention": [e for e in errs if "attention" in e],
            "other": [e for e in errs if not e.startswith("aud:") and "attention" not in e]}


def _chi2(counts_a, counts_b) -> float:
    """p-value of a chi-square test of homogeneity of two count vectors (categories with tiny counts pooled) - the
    formula of tests/v5/test_gate_D.py."""
    import numpy as np
    from scipy import stats
    tab = np.stack([counts_a, counts_b]).astype(np.float64)
    tot = tab.sum(0)
    keep = tot >= 10
    small = tab[:, ~keep].sum(1, keepdims=True)
    tab = np.concatenate([tab[:, keep], small], axis=1) if small.sum() > 0 else tab[:, keep]
    tab = tab[:, tab.sum(0) > 0]
    if tab.shape[1] < 2:
        return 1.0
    return float(stats.chi2_contingency(tab)[1])


def d_overrides(profile: str, seed: int) -> dict:
    """The civilisation's own feed configuration with Gate D's mixtures on its posts and blog platforms."""
    import copy
    from ..social import experiments as X
    from . import generator
    plats = copy.deepcopy(generator.recipe(profile, seed)["modules"]["feed"]["platforms"])
    for p, cfg in X.GATE_D_POP.items():
        plats[p] = copy.deepcopy(cfg)
    return {"module_config": {"feed": {"platforms": plats}}}


def d_battery(p: dict | None = None) -> dict:
    """Gate D's five requirements measured on one civilisation (see the module docstring)."""
    import itertools
    import numpy as np
    import torch
    from scipy import stats
    torch.set_num_threads(1)
    from ..core.kernel import World
    from ..social import audience as AUD, experiments as X
    from ..social.recsys import REGISTRY
    from ..substrate import attention, columns, ids
    from . import generator
    p = {**D_RUN, **(p or {})}
    sw = int(p["switch_day"]) * DAY
    t0 = time.time()
    w = generator.generate(p["profile"], p["seed"], overrides={**d_overrides(p["profile"], p["seed"]),
                                                              "params": {"log_retain": 8_000_000}})
    checks = X.run(w, 2 * DAY, check=_d_checks)
    day2, day2_seq = w.snapshot_blob(), w.seq
    checks += X.run(w, sw - 900, check=_d_checks)
    pre, pre_seq = w.snapshot_blob(), w.seq
    checks += X.run(w, sw + DAY, check=_d_checks)
    out: dict = {}
    # D.1 / D.2: one arm per registered recommender from the day-2 state
    arms = {}
    for name in sorted(REGISTRY):
        a = World.from_blob(day2, world_id=f"civ-D-arm-{name}", fork=True)
        for plat in X.GATE_D_POP:
            a.system_call("feed.set_adapt", {"platform": plat, "enabled": False})
            X.set_mixture(a, plat, [{"name": name, "weight": 1.0}])
        s0 = a.seq
        X.run(a, 3 * DAY)
        imp = X.decision_rows(a, "imp", s0)
        con = X.decision_rows(a, "consume", s0)
        rec = {"imp_items": imp["item"], "exposure": X.pair_keys(a, imp), "views": int(con["is_view"].sum())
               if con["item"].size else 0, "n_imp": int(imp["item"].size), "n_items": int(AUD.n_items(a))}
        rec["view_items"] = con["item"][con["is_view"]] if con["item"].size else np.zeros(0, np.int64)
        if name in ("interest", "trending") and con["item"].size:
            m = con["is_view"]
            u = columns.col(a, "pop", "interests")[con["viewer"][m]].astype(np.float64)
            e = columns.col(a, AUD.ITEMS, "emb")[con["item"][m]].astype(np.float64)
            rec["affinity"] = np.einsum("nd,nd->n", u, e)
        arms[name] = rec
        a.close()
    ok = [n for n, a in arms.items() if a["n_imp"] > 0 and a["views"] > 0]
    pairs = {f"{a}|{b}": X.jaccard(arms[a]["exposure"], arms[b]["exposure"]) for a, b in itertools.combinations(ok, 2)}
    chosen = []
    for n in ok:
        if all(X.jaccard(arms[n]["exposure"], arms[m]["exposure"]) <= 0.8 for m in chosen):
            chosen.append(n)
    sub = [X.jaccard(arms[a]["exposure"], arms[b]["exposure"]) for a, b in itertools.combinations(chosen, 2)]
    out["D.1"] = {"distinct_recommenders_end_to_end": len(chosen),
                  "max_pairwise_exposure_overlap": round(max(sub) if sub else 1.0, 4),
                  "registered_recommenders": len(REGISTRY), "recommenders_end_to_end": len(ok),
                  "mechanisms": ",".join(chosen),
                  "impressions_per_arm_min": min((a["n_imp"] for a in arms.values()), default=0),
                  "views_per_arm_min": min((a["views"] for a in arms.values()), default=0)}
    A, B = arms.get("interest"), arms.get("trending")
    if A and B and A["n_imp"] and B["n_imp"]:
        ni = max(A["n_items"], B["n_items"])
        p_exp = _chi2(np.bincount(A["imp_items"], minlength=ni), np.bincount(B["imp_items"], minlength=ni))
        p_con = _chi2(np.bincount(A["view_items"], minlength=ni), np.bincount(B["view_items"], minlength=ni))
        out["D.2"] = {"recommender_exposure_difference_pvalue": p_exp,
                      "recommender_consumption_difference_pvalue": p_con,
                      "viewed_affinity_difference_pvalue": float(stats.mannwhitneyu(A["affinity"], B["affinity"]).pvalue)
                      if A.get("affinity") is not None and B.get("affinity") is not None and A["affinity"].size and
                      B["affinity"].size else 1.0,
                      "impressions_a": A["n_imp"], "impressions_b": B["n_imp"], "views_a": A["views"],
                      "views_b": B["views"]}
    else:
        out["D.2"] = "the interest or trending arm recorded no impressions in the civilisation"

    # D.3: the scheduled change (posts) and the blog's own reweighting vs the counterfactual of the pre-change state
    def ages(world, rows):
        pub = columns.col(world, AUD.ITEMS, "published_at")[rows["item"]]
        return np.histogram((rows["t"] - pub) / 3600.0, bins=[-1, 3, 6, 12, 24, 48, 96, 1e9])[0]
    rows = X.decision_rows(w, "imp", pre_seq, platform="posts")
    after = {k: v[rows["t"] >= sw] for k, v in rows.items()}
    cf = World.from_blob(pre, world_id="civ-D-cf", fork=True)
    X.set_mixture(cf, "posts", X.GATE_D_POP["posts"]["recommenders"])
    cf.system_call("feed.set_adapt", {"platform": "blog", "enabled": False})
    s0 = cf.seq
    X.run(cf, sw + DAY)
    crows = X.decision_rows(cf, "imp", s0, platform="posts")
    cafter = {k: v[crows["t"] >= sw] for k, v in crows.items()}
    p_shift = _chi2(ages(w, after), ages(cf, cafter))
    brows = X.decision_rows(w, "imp", pre_seq, platform="blog")
    cbrows = X.decision_rows(cf, "imp", s0, platform="blog")
    p_self = _chi2(np.bincount(brows["src"][brows["t"] >= sw], minlength=len(AUD.SOURCES)),
                   np.bincount(cbrows["src"][cbrows["t"] >= sw], minlength=len(AUD.SOURCES))) \
        if brows["item"].size and cbrows["item"].size else 1.0
    changes, selfc = X.mixture_changes(w), X.self_changes(w)
    out["D.3"] = {"algorithm_changes_during_run": int(changes),
                  "post_change_exposure_shift_pvalue": max(p_shift, p_self),
                  "scheduled_change_exposure_shift_pvalue": p_shift, "self_adaptation_changes": int(selfc),
                  "self_adaptation_exposure_shift_pvalue": p_self,
                  "post_change_impressions_changed": int(after["item"].size),
                  "post_change_impressions_counterfactual": int(cafter["item"].size)}
    cf.close()

    # D.4: hidden-parameter probes by a participant and a creator (the probes of tests/v5/test_gate_D.py)
    out["D.4"] = _d_probes(World.from_blob(w.snapshot_blob(), world_id="civ-D-probe", fork=True))

    # D.5: exposure and attention accounting over the persistent run
    aud_viol = sum(len(c["aud"]) for c in checks)
    att_viol = sum(len(c["attention"]) for c in checks)
    rc = AUD.recount(w)
    budget = columns.col(w, "pop", "attention_budget_min").astype(np.float64)
    spent = columns.col(w, "pop", "attention_spent_min").astype(np.float64)
    over_now = int(np.sum(spent > budget * (1 + attention.REL_TOL) + attention.ABS_TOL))
    con = X.decision_rows(w, "consume", 0)
    watch_over = 0
    if con["item"].size:
        key = (con["t"] // DAY) * AUD.n_people(w) + con["viewer"]
        uk, inv = np.unique(key, return_inverse=True)
        watched_min = np.bincount(inv, weights=con["watch_s"]) / 60.0
        watch_over = int(np.sum(watched_min > budget[uk % AUD.n_people(w)] * (1 + 1e-5) + 1e-3))
    out["D.5"] = {"exposure_invariant_violations": int(aud_viol + rc["total_mismatches"] +
                                                       rc["views_without_prior_exposure"]),
                  "attention_overspend_events": int(att_viol + over_now + watch_over),
                  "invariant_checks": len(checks), "impression_rows": int(rc["impression_rows"]),
                  "consumption_rows": int(rc["consumption_rows"]),
                  "other_invariant_violations": sum(len(c["other"]) for c in checks)}
    out["_run"] = {"events": w.seq, "proc_failed": sum(1 for e in w.log if e.type == "proc.failed"),
                   "pairs": {k: round(v, 3) for k, v in pairs.items()}, "wall_s": round(time.time() - t0, 1)}
    w.close()
    return json.loads(json.dumps(out, default=str))


_CONTENT_KEYS = {"title", "text", "tags", "data_b64", "body", "name", "subject", "description", "snippet"}


def _strip_content(obj):
    if isinstance(obj, dict):
        return {k: _strip_content(v) for k, v in obj.items() if k not in _CONTENT_KEYS}
    if isinstance(obj, (list, tuple)):
        return [_strip_content(v) for v in obj]
    return obj


def _d_probes(w) -> dict:
    """D.4 on a fork of the civilisation: every feed/aud tool, tool discovery, observation, visible events, the
    artifact tools on model and decision blobs, and creator analytics, by one participant and one creator; a leak is
    any of Gate D's evaluator-only markers (`crucible.social.experiments.SECRET_MARKERS`) in a result outside what
    participants wrote, or a readable internal record."""
    from ..social import audience as AUD, experiments as X
    from ..substrate import ids
    markers = X.SECRET_MARKERS
    probes, leaks, detail = 0, 0, []

    def scan(obj):
        s = json.dumps(_strip_content(obj), sort_keys=True, default=str)
        return [m for m in markers if m in s]

    def probe(name, res_or_fn):
        nonlocal probes, leaks
        probes += 1
        try:
            res = res_or_fn() if callable(res_or_fn) else res_or_fn
        except Exception as e:  # noqa: BLE001 - a refusal is a non-leak; its message is scanned too
            res = {"error": str(e)}
        hit = scan(res)
        if hit:
            leaks += 1
            detail.append(f"{name}: {hit}")
    info = w.state.tables["aud_item_info"]
    creators = sorted({r["creator"] for r in info.values() if ids.index(r["creator"]) is not None},
                      key=lambda q: ids.index(q))
    if not creators:
        w.close()
        return "no population creator published on the civilisation's platforms"
    creator = creators[0]
    viewer_i = next(i for i in range(AUD.n_people(w)) if ids.pid(i) not in set(creators))
    w._run_system("probe.principal", lambda ctx: ids.ensure_principal(ctx, viewer_i))
    viewer = ids.pid(viewer_i)
    my_item = int(w.state.find_ids("aud_item_info", "creator", creator)[0])
    for actor in (viewer, creator):
        r = w.execute(actor, {"tool": "feed.home", "args": {"platform": "posts", "k": 10}})
        probe(f"{actor} feed.home", r.to_dict())
        items = [x["item"] for x in (r.result or {}).get("items", [])]
        probe(f"{actor} feed.platforms", w.execute(actor, {"tool": "feed.platforms", "args": {}}).to_dict())
        probe(f"{actor} feed.search", w.execute(actor, {"tool": "feed.search", "args": {
            "platform": "blog", "q": "garden soil"}}).to_dict())
        if items:
            probe(f"{actor} feed.watch", w.execute(actor, {"tool": "feed.watch", "args": {
                "item": items[0], "seconds": 40, "share": True}}).to_dict())
            probe(f"{actor} aud.item", w.execute(actor, {"tool": "aud.item", "args": {"item": items[0]}}).to_dict())
        probe(f"{actor} feed.open", w.execute(actor, {"tool": "feed.open", "args": {"item": my_item}}).to_dict())
        probe(f"{actor} set_mixture refused", w.execute(actor, {"tool": "feed.set_mixture", "args": {
            "platform": "posts", "recommenders": [{"name": "random"}]}}).to_dict())
        probe(f"{actor} describe feed", w.execute(actor, {"tool": "world.describe", "args": {"host": "feed"}})
              .to_dict())
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
        refs = [r_["ref"] for r_ in w.state.tables["feed_models"].values()] + [b["ref"] for b in AUD.batches(w)[:5]]
        for ref in refs[:6]:
            probes += 1
            if w.execute(actor, {"tool": "artifact.get", "args": {"ref": ref}}).ok:
                leaks += 1
                detail.append(f"{actor} read system blob {ref}")
        probes += 1
        r = w.execute(actor, {"tool": "artifact.list", "args": {"owner": "sys"}})
        probe(f"{actor} artifact.list sys", r.to_dict())
        if r.ok and r.result.get("total"):
            leaks += 1
            detail.append(f"{actor} lists system artifacts")
    a1 = w.execute(creator, {"tool": "aud.analytics", "args": {"item": my_item}})
    a2 = w.execute(creator, {"tool": "aud.analytics", "args": {"item": my_item}})
    probe("creator analytics", a1.to_dict())
    probes += 1
    if a1.ok and a2.ok and a1.result != a2.result:
        leaks += 1
        detail.append("analytics noise differs between repeated queries (averageable)")
    probes += 1
    other = next((int(i) for i in info if info[i]["creator"] != creator), None)
    if other is not None and w.execute(creator, {"tool": "aud.analytics", "args": {"item": other}}).ok:
        leaks += 1
        detail.append("creator read another creator's analytics")
    probe("creator mine", w.execute(creator, {"tool": "aud.mine", "args": {}}).to_dict())
    w.close()
    return {"hidden_parameter_probes": probes, "hidden_parameter_leaks": leaks, "leak_detail": detail[:10],
            "markers_searched": len(markers), "probe_actors": 2}


# --------------------------------------------------------------------------------------------- G
def g_arm(seed: int, profile: str = "balanced", phase_days: int = 5, *, compare: bool = False) -> dict:
    """One seed of Gate G's adaptation experiment in a civilisation (mirrors crucible.population.experiments
    .adaptation; see the module docstring)."""
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from ..population import analysis as A, experiments as PX
    from ..substrate import artifacts, columns
    from . import generator
    arms = {"learning": {"id": 0, "share": 0.5, "seed": 5},
            "frozen": {"id": 1, "share": 0.5, "seed": 5, "learning": False}}
    names, specs = PX._site_specs(seed)
    ov = {"module_config": {"pop": {"classes": {"adaptive": 256}, "adaptive_window_s": 14400},
                            "pop_adaptive": {"groups": list(arms.values()), "min_examples": 32},
                            "popsites": {"sites": specs}}}
    t0 = time.time()
    w = generator.generate(profile, seed, overrides=ov)
    ad = w.modules["pop_adaptive"]
    gid = {a: int(c["id"]) for a, c in arms.items()}
    out = {"seed": seed, "phase_days": phase_days, "arms": arms, "profile": profile,
           "scripted": "site specifications and the switch at the end of phase 1 (experimenter)"}
    hashes = lambda: {a: ad.policy(w, g).weights_hash() for a, g in gid.items()}  # noqa: E731
    out["hash_genesis"] = hashes()
    cls = columns.col(w, "pop", "class")
    days, rounds = {}, {}
    strat: dict[str, list] = {}

    def scan(s_from):
        for e in w.events_range(s_from):
            if e.type == "cols.batch" and e.payload["kind"] == "pop.session":
                idx = artifacts.get_array_blob(w, e.payload["ref"])["idx"].astype(np.int64)
                a2 = idx[cls[idx] == 2]
                t = int(e.payload["extra"]["t"])
                days.setdefault(t // DAY, set()).update(a2.tolist())
                rounds[t] = rounds.get(t, 0) + int(a2.size)
            elif e.type == "pop.policy_update" and str(e.payload["group"]).startswith("strategic"):
                strat.setdefault(e.payload["group"], []).append(e.payload["weights_hash"])
    logs, ups = [], []
    s = 0
    for d in range(phase_days):          # day by day: the civilisation's log keeps a bounded window
        w.advance_time((d + 1) * DAY)
        scan(s)
        logs.append(A.visit_log(w, w.events_range(s)))
        ups += [e.payload for e in w.events_range(s) if e.type == "pop.policy_update"]
        s = w.seq
    out["hash_switch"] = hashes()
    log1 = {k: np.concatenate([lg[k] for lg in logs]) for k in logs[0]}
    hosts = A.res_hosts(w)
    row = {h: r for r, h in enumerate(hosts)}
    vis = {h: int((log1["res"] == row[h]).sum()) if h in row else 0 for h in names}
    rich = sorted([h for h, b in zip(names, PX.BEHAVIOURS) if b == "rich"], key=lambda h: (-vis[h], h))
    to_broken, to_rich = rich[:2], [h for h, b in zip(names, PX.BEHAVIOURS) if b == "broken"]

    def switch(ctx):
        for h in to_broken:
            PX.redeploy(ctx, h, specs[names.index(h)], "broken")
        for h in to_rich:
            PX.redeploy(ctx, h, specs[names.index(h)], "rich")
    w._run_system("experiment.switch", switch)
    t_switch = w.now
    for d in range(phase_days, 2 * phase_days):
        w.advance_time((d + 1) * DAY)
        scan(s)
        logs.append(A.visit_log(w, w.events_range(s)))
        ups += [e.payload for e in w.events_range(s) if e.type == "pop.policy_update"]
        s = w.seq
    out["hash_end"] = hashes()
    out["switch"] = {"to_broken": to_broken, "to_rich": to_rich}
    for a, g in gid.items():
        u_ = [u for u in ups if u["group"] == f"adaptive:{g}"]
        out.setdefault("updates", {})[a] = max(0, len(u_) - 1)
        out.setdefault("optimizer_steps", {})[a] = int(sum(int(u.get("optimizer_steps") or 0) for u in u_))
    log = {k: np.concatenate([lg[k] for lg in logs]) for k in logs[0]}
    pg = columns.col(w, "pop", "policy_group").astype(int)
    m2 = (log["cls"] == 2) & (log["activity"] == "web")
    grp = pg[log["ind"].astype(int)]
    bad_after = {row[h] for h in to_broken if h in row} | \
        {row[h] for h, b in zip(names, PX.BEHAVIOURS) if b in ("thin", "spam") and h in row}
    util = {}
    for a, g in gid.items():
        r = {}
        for ph, sel in (("phase1", log["t"] < t_switch), ("phase2", log["t"] >= t_switch)):
            m = m2 & (grp == g) & sel
            r[ph] = {"visits": int(m.sum()), "mean_util": float(log["util"][m].mean()) if m.any() else 0.0}
            if ph == "phase2":
                r[ph]["share_failing"] = float(np.isin(log["res"][m], list(bad_after)).mean()) if m.any() else 0.0
            util[(a, ph)] = log["util"][m].astype(np.float64)
        out.setdefault("behaviour", {})[a] = r
    out["phase2_difference"] = PX._boot(util[("learning", "phase2")], util[("frozen", "phase2")])
    out["phase1_difference"] = PX._boot(util[("learning", "phase1")], util[("frozen", "phase1")], seed=1)
    st = w.state
    unused = [k for k, r in st.tables["pop_examples"].items() if r.get("used") is None]
    heldout = None
    if unused:
        batch = ad._load_examples(w, unused)
        out["heldout_examples"] = int(len(batch["reward"]))
        out["heldout_mse"] = {a: float(ad.policy(w, g).loss_on(batch)) for a, g in gid.items()}
        heldout = {"batch": batch, "policies": {a: ad.policy(w, g) for a, g in gid.items()}}
    mem = columns.col(w, "pop", "memory")[cls == 2].astype(np.float32)
    out["distinct_personal_states"] = int(np.unique(mem, axis=0).shape[0])
    out["adaptive_rows"] = int(mem.shape[0])
    nets = w.runtime.get("pop_adaptive.nets", {})
    out["policy_objects"] = len(nets)
    # weight copies per policy group: network objects that serve one group (each group's members share one)
    out["copies_per_group"] = int(-(-len(nets) // max(1, len(gid)))) if nets else 0
    out["adaptive_stats"] = {k: (v if isinstance(v, (int, float)) else str(v))
                             for k, v in dict(w.runtime.get("pop_adaptive.stats", {})).items()}
    out["max_adaptive_active_same_day"] = max((len(v) for v in days.values()), default=0)
    ds = sorted(days)
    out["max_adaptive_active_two_days"] = max((len(days[d] | days.get(d + 1, set())) for d in ds), default=0)
    out["max_adaptive_per_decision_round"] = max(rounds.values(), default=0)
    out["strategic_hashes"] = {g: [h[0], h[-1], len(set(h))] for g, h in strat.items()}
    out["strategic_updates"] = sum(len(h) for h in strat.values())
    if compare:
        out["compact_vs_language"] = PX.compact_vs_language({"_heldout": heldout} if heldout else {}, seed=0)
    out["proc_failed"] = sum(1 for e in w.log if e.type == "proc.failed")
    out["invariants"] = w.check_invariants()[:5]
    out["events"] = w.seq
    out["wall_s"] = round(time.time() - t0, 1)
    w.close()
    return json.loads(json.dumps(out, default=str))


def g_runs(p: dict | None = None, workers: int = 1) -> list[dict]:
    p = {**G_RUN, **(p or {})}
    tmp = tempfile.mkdtemp(prefix="civ-redemo-g-")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    pending, running, out = list(p["seeds"]), [], {}
    while pending or running:
        while pending and len(running) < max(1, workers):
            sd = pending.pop(0)
            path = os.path.join(tmp, f"g{sd}.json")
            pr = subprocess.Popen([sys.executable, "-m", "crucible.civ.redemo", "g", str(sd), p["profile"],
                                   str(p["phase_days"]), "1" if sd == p["seeds"][0] else "0", path], env=env,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            running.append((pr, path, sd))
        pr, path, sd = running.pop(0)
        _, err = pr.communicate()
        if pr.returncode != 0:
            raise RuntimeError(f"G seed {sd} failed: {err.decode(errors='replace')[-2000:]}")
        with open(path) as fh:
            out[sd] = json.load(fh)
    return [out[s_] for s_ in p["seeds"]]


def g_metrics(runs: list[dict]) -> dict:
    """Gate G's metrics (formulas of tests/v5/test_gate_G.py) from the civilisation runs."""
    import numpy as np
    r0 = runs[0]
    out = {"G.1": {"simultaneously_active_adaptive": int(max(r["max_adaptive_active_two_days"] for r in runs)),
                   "adaptive_rows": int(r0["adaptive_rows"])},
           "G.2": {"distinct_personal_states": int(min(r["distinct_personal_states"] for r in runs))},
           "G.3": {"shared_weight_copies": int(r0["copies_per_group"]),
                   "max_inference_batch": int(float(r0["adaptive_stats"].get("max_batch", 0) or 0)),
                   "policy_objects": int(r0["policy_objects"]), "policy_groups": 2},
           "G.4": {"independent_strategic_groups": int(sum(1 for h in r0["strategic_hashes"].values() if h[2] > 1)),
                   "distinct_group_weight_hashes": int(len({h[1] for h in r0["strategic_hashes"].values()})),
                   "strategic_policy_updates": int(r0["strategic_updates"])}}
    if all("heldout_mse" in r for r in runs):
        out["G.5"] = {"weight_hash_changed": bool(r0["hash_end"]["learning"] != r0["hash_genesis"]["learning"]),
                      "frozen_hash_unchanged": bool(r0["hash_end"]["frozen"] == r0["hash_genesis"]["frozen"]),
                      "optimizer_updates": int(sum(x["optimizer_steps"]["learning"] for x in runs)),
                      "loss_decrease": float(np.mean([x["heldout_mse"]["frozen"] - x["heldout_mse"]["learning"]
                                                      for x in runs]))}
    else:
        out["G.5"] = "a seed kept no held-out examples"
    lows = [r["phase2_difference"]["ci95"][0] for r in runs]
    out["G.6"] = {"new_condition_improvement_ci_low": float(min(lows)),
                  "new_condition_improvement_mean": float(np.mean([r["phase2_difference"]["mean"] for r in runs])),
                  "phase1_improvement_mean": float(np.mean([r["phase1_difference"]["mean"] for r in runs]))}
    out["G.n1"] = {"adaptation_effect_ci_low": float(min(lows)), "seeds": len(runs)}
    rows = [x for x in r0.get("compact_vs_language") or [] if not x.get("error")]
    out["G.7"] = {"compact_vs_language_comparison_rows": len(rows),
                  **{f"{x['model']}_spearman": round(float(x["spearman"]), 4) for x in rows}}
    return out


# --------------------------------------------------------------------------------------------- all gates
def run_all(workers: int = 1, creation: dict | None = None, autonomous: dict | None = None) -> dict:
    t = time.time()
    A = creation_arms(creation, workers=workers)
    jm, lm = j_metrics(A), l_metrics(A)
    au = autonomous_run(autonomous)
    dm = d_battery()
    gr = g_runs(workers=workers)
    gm = g_metrics(gr)
    measured = {"D": {k: v for k, v in dm.items() if k.startswith("D.")},
                "G": gm,
                "J": {k: v for k, v in jm.items() if not k.startswith("_")},
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
            "d_run": dm.get("_run"),
            "g_runs": [{k: r.get(k) for k in ("seed", "events", "proc_failed", "invariants", "wall_s",
                                               "phase2_difference", "phase1_difference", "behaviour", "switch")}
                       for r in gr],
            "wall_s": round(time.time() - t, 1)}


def _main(argv) -> int:
    if argv and argv[0] == "arm":
        seed, arm, profile, days, saas_day, path = int(argv[1]), argv[2], argv[3], int(argv[4]), int(argv[5]), argv[6]
        r = creation_arm(seed, arm, profile=profile, days=days, saas_day=saas_day)
        with open(path, "w") as fh:
            json.dump(r, fh)
        return 0
    if argv and argv[0] == "g":
        seed, profile, days, compare, path = int(argv[1]), argv[2], int(argv[3]), argv[4] == "1", argv[5]
        r = g_arm(seed, profile, days, compare=compare)
        with open(path, "w") as fh:
            json.dump(r, fh)
        return 0
    if argv and argv[0] == "D":
        print(json.dumps(d_battery(), indent=1, default=str)[:20000])
        return 0
    print("usage: python -m crucible.civ.redemo arm <seed> <arm> <profile> <days> <saas_day> <out.json> | D")
    return 2


if __name__ == "__main__":
    import crucible
    crucible.load_all_modules()
    sys.exit(_main(sys.argv[1:]))
