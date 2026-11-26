"""Gate O / P / SP-10.x measurements on generated civilisations (the gate tests call these; so can a person:
`python -m crucible.civ.gates <what>`).

Expensive results (counterfactual experiments, long-horizon suites) are cached on disk under
`$CRUCIBLE_CIV_CACHE` (default `<tmp>/crucible-civ-cache`) keyed by a digest of every source file of this package
and the parameters, so two gate tests that need the same experiment run it once; any code change invalidates the
cache. Nothing is cached across code versions and nothing is ever read from a fixed result file.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time

WORKERS_ENV = "CRUCIBLE_CIV_WORKERS"
CACHE_ENV = "CRUCIBLE_CIV_CACHE"
INTEGRATION_RUN = {"profile": "balanced", "seed": 7, "days": 5}
EFFECT_SEEDS = (11, 12, 13)
CHAIN_SEEDS = (21, 22, 23)


def workers() -> int:
    return max(1, int(os.environ.get(WORKERS_ENV, "1")))


def code_digest() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    roots = [here, os.path.join(os.path.dirname(here), "economy", "industry.py")]
    for r in roots:
        files = [r] if r.endswith(".py") else sorted(os.path.join(r, f) for f in os.listdir(r) if f.endswith(".py"))
        for f in files:
            with open(f, "rb") as fh:
                h.update(f.encode() + fh.read())
    return h.hexdigest()[:16]


def _cache_path(kind: str, params: dict) -> str:
    d = os.environ.get(CACHE_ENV) or os.path.join(tempfile.gettempdir(), "crucible-civ-cache")
    os.makedirs(d, exist_ok=True)
    key = hashlib.sha256(json.dumps({"kind": kind, "params": params, "code": code_digest()},
                                    sort_keys=True).encode()).hexdigest()[:20]
    return os.path.join(d, f"{kind}-{key}.json")


def cached(kind: str, params: dict, fn):
    path = _cache_path(kind, params)
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    out = fn()
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, default=str)
    os.replace(tmp, path)
    return out


# --------------------------------------------------------------------------------------------- O.1-O.4, O.n1
def integration(profile: str | None = None, seed: int | None = None, days: int | None = None) -> dict:
    """One civilisation run; integration metrics over its whole event log."""
    import torch
    torch.set_num_threads(1)
    from . import chains, generator, integration as I
    p = dict(INTEGRATION_RUN)
    p.update({k: v for k, v in (("profile", profile), ("seed", seed), ("days", days)) if v is not None})

    def run():
        t = time.time()
        w = generator.generate(p["profile"], p["seed"])
        w.advance_time(86400)                 # warm-up day: genesis, site installation, first sessions
        since = w.seq
        w.advance_time(int(p["days"]) * 86400)
        pv = chains.build(w, 0)
        r = generator.recipe(p["profile"], p["seed"])
        out = {"params": p, "events": w.seq, "modules": sorted(w.modules), "pending": r["pending"],
               "excluded": sorted(r["excluded"]), "measured_after_seq": since,
               "subsystems": I.subsystems(w, pv, since), "cross": I.cross_system(pv, since),
               "shared": I.modules_without_shared_state(w, pv, since), "duplicates": I.duplicate_facts(w),
               "identity": I.identity_violations(w), "ledger": I.ledger_violations(w),
               "invariants": w.check_invariants()[:20], "wall_s": round(time.time() - t, 1)}
        w.close()
        return out
    return cached("integration", p, run)


# --------------------------------------------------------------------------------------------- O.5
def effects(seeds=EFFECT_SEEDS, names=None) -> dict:
    from . import counterfactual as CF, experiments as X
    names = list(names or X.EFFECT_EXPERIMENTS)
    out = {}
    for n in names:
        out[n] = _safe(n, "effect", seeds)
    confirmed = sorted({X.EXPERIMENTS[n].category for n in names if out[n]["confirmed"]})
    from .couplings import CATALOGUE
    pending = sorted({c.category for c in CATALOGUE.values() if c.pending})
    return {"cross_domain_effect_types": len(confirmed), "confirmed_categories": confirmed,
            "pending_categories": pending, "experiments": out}


def _safe(name: str, kind: str, seeds) -> dict:
    """Run (or load from the cache) one experiment; a crash is recorded as an unconfirmed result with its error."""
    from . import counterfactual as CF, experiments as X
    try:
        return cached(kind, {"name": name, "seeds": list(seeds)},
                      lambda: _strip(CF.run(name, list(seeds), workers=workers())))
    except Exception as e:  # noqa: BLE001 - reported, never hidden
        exp = X.EXPERIMENTS[name]
        return {"experiment": name, "category": exp.category, "primary": exp.primary, "confirmed": False,
                "error": f"{type(e).__name__}: {e}"[:3000],
                "effects": {a: {exp.primary: {"effect_mean": float("nan"), "ci95": [float("nan")] * 2,
                                              "reshuffle_noise_rms": float("nan"), "confirmed": False},
                                "confirmed": False} for a in exp.interventions}}


def _strip(r: dict) -> dict:
    r = dict(r)
    per = r.pop("per_seed", [])
    r["per_seed_chains"] = [p.get("chains", []) for p in per]
    r["per_seed_wall"] = [p.get("wall_s") for p in per]
    return r


# --------------------------------------------------------------------------------------------- O.6, SP-10.3
CHAIN_ARMS = {
    # experiment -> arm -> (what the arm removes, chain class, {measure: subsystem it stands for})
    "agent_shop": {
        "no_agent": ("the agent's web shop (repository, deployment, posts)", "agent_artifact",
                     {"agent_visitors": "population", "agent_revenue": "finance",
                      "provider_cu": "computing_infrastructure", "provider_kwh": "energy"}),
        "no_post": ("the agent's forum posts (the shop stays)", "social",
                    {"agent_visitors": "population", "agent_revenue": "finance",
                     "provider_cu": "computing_infrastructure", "provider_kwh": "energy"})},
    "agent_science": {
        "no_paper": ("the agent's paper (its measurements stay)", "technical",
                     {"adopted": "industry", "true_quality": "industry", "bound_yield": "industry",
                      "goods_bought": "population"})},
}
ROOT_SUBSYSTEM = {"agent_artifact": "software_development", "social": "search_information",
                  "technical": "scientific_research"}


def chain_results(seeds=CHAIN_SEEDS) -> dict:
    from . import counterfactual as CF
    chains_out = []
    raw = {}
    for exp, arms in CHAIN_ARMS.items():
        r = _safe(exp, "chain", seeds)
        raw[exp] = r
        mined = [c for per in r.get("per_seed_chains", []) for c in per]
        for arm, (what, cls, msub) in arms.items():
            eff = r["effects"][arm]
            confirmed_subs = sorted({msub[m] for m, v in eff.items() if isinstance(v, dict) and v.get("confirmed")
                                     and m in msub})
            stages = [ROOT_SUBSYSTEM[cls]] + [s for s in confirmed_subs if s != ROOT_SUBSYSTEM[cls]]
            # provenance: a mined chain of the agent with this class in every seed's reference arm
            support = [c for c in mined if c["actor_kind"] == "agent" and c.get(cls)]
            chains_out.append({"experiment": exp, "arm": arm, "removed": what, "class": cls,
                               "confirmed_stages": stages, "n_confirmed_stages": len(stages),
                               "mined_in_reference_arm": len(support),
                               "max_mined_stages": max((c["n_stages"] for c in support), default=0),
                               "scripted_root": True,
                               "effects": {m: {k: v[k] for k in ("effect_mean", "ci95", "reshuffle_noise_rms",
                                                                 "confirmed")}
                                           for m, v in eff.items() if isinstance(v, dict)}})
    good = [c for c in chains_out if c["n_confirmed_stages"] >= 3 and c["mined_in_reference_arm"] > 0]
    return {"multi_stage_chains": len(good), "min_chain_stages": min((c["n_confirmed_stages"] for c in good),
                                                                     default=0),
            "chains_with_agent_artifact": sum(1 for c in good if c["class"] == "agent_artifact"),
            "chains_with_social_activity": sum(1 for c in good if c["class"] == "social"),
            "chains_with_technical_activity": sum(1 for c in good if c["class"] == "technical"),
            "chains": chains_out, "experiments": raw}


# --------------------------------------------------------------------------------------------- O.7, SP-10.1
def structure(seed: int = 5) -> dict:
    """Fingerprints of every profile's BUILT world (after genesis and the first simulated hour)."""
    import torch
    torch.set_num_threads(1)
    from . import generator, integration as I, profiles as P

    def run():
        fps = {}
        for name in sorted(P.PROFILES):
            w = generator.generate(name, seed)
            w.advance_time(3600)
            fps[name] = I.fingerprint(w)
            w.close()
        names = sorted(fps)
        same = [[a, b] for i, a in enumerate(names) for b in names[i + 1:] if fps[a] == fps[b]]
        dist = {f"{a}|{b}": round(I.structural_distance(fps[a], fps[b]), 3)
                for i, a in enumerate(names) for b in names[i + 1:]}
        distinct = len({json.dumps(f, sort_keys=True) for f in fps.values()})
        return {"profiles": names, "fingerprints": fps, "pairs_identical": same, "distances": dist,
                "structurally_distinct_profiles": distinct,
                "part_10_1": {n: P.PROFILES[n].part_10_1 for n in names}}
    return cached("structure", {"seed": seed}, run)


# --------------------------------------------------------------------------------------------- P
P_HORIZONS = (30, 90, 180, 365)


def long_horizon(horizons=P_HORIZONS, seeds_per_horizon: int = 2) -> dict:
    from . import longrun

    def run():
        reps = longrun.suite(horizons, seeds_per_horizon, workers=workers(),
                             log=lambda m: print(m, file=sys.stderr, flush=True))
        for r in reps:
            r.pop("daily", None)
        return {"reports": reps}
    res = cached("longrun", {"horizons": list(horizons), "seeds": seeds_per_horizon,
                             "profile": longrun.GATE_P_PROFILE}, run)
    return {"metrics": p_metrics(res["reports"], horizons), "reports": res["reports"]}


def p_metrics(reps: list[dict], horizons=P_HORIZONS) -> dict:
    """Gate P metrics from long-run reports (crucible.civ.longrun.run_horizon)."""
    by_h = {h: [r for r in reps if r["days"] == h] for h in horizons}

    def ok(r):
        s = r.get("summary") or {}
        return not r.get("failed") and not s.get("failed")
    m = {f"runs_{h}_days": sum(1 for r in by_h[h] if ok(r)) for h in horizons}
    m["failed_runs"] = sum(1 for r in reps if not ok(r))
    S = [r.get("summary") or {} for r in reps]
    m["unexplained_invariant_violations"] = sum(s.get("unexplained_invariant_violations", 0) for s in S) + \
        sum(1 for r in reps if r.get("failed"))
    m["snapshot_resume_mismatches"] = sum(s.get("resume_mismatches", 0) for s in S)
    m["snapshot_resume_checks"] = sum(s.get("resume_checks", 0) for s in S)
    m["replay_mismatches"] = sum((s.get("replay_mismatches") or 0) for s in S) + \
        sum(1 for s in S if not s.get("replay_segments"))
    m["event_accounting_mismatches"] = sum(s.get("accounting_mismatches", 0) for s in S)
    m["min_late_over_early_economic_activity"] = min((s.get("late_over_early_economic", 0.0) for s in S),
                                                     default=0.0)
    m["min_late_over_early_social_activity"] = min((s.get("late_over_early_social", 0.0) for s in S), default=0.0)
    m["independent_seeds_per_horizon"] = min((len({r["seed"] for r in by_h[h] if ok(r)}) for h in horizons),
                                             default=0)
    return m


def _main(argv):
    import crucible
    crucible.load_all_modules()
    what = argv[0] if argv else "integration"
    if what == "experiment":          # one experiment into the cache: experiment <name> effect|chain
        from . import experiments as X
        kind = argv[2] if len(argv) > 2 else ("chain" if X.EXPERIMENTS[argv[1]].chain else "effect")
        seeds = CHAIN_SEEDS if kind == "chain" else EFFECT_SEEDS
        r = _safe(argv[1], kind, seeds)
        print(json.dumps({"experiment": argv[1], "confirmed": r.get("confirmed"), "error": r.get("error"),
                          "effects": r.get("effects")}, indent=1, default=str)[:20000])
        return 0
    fn = {"integration": integration, "effects": effects, "chains": chain_results, "structure": structure,
          "long": long_horizon}[what]
    r = fn()
    if what in ("effects", "chains"):
        r = {k: v for k, v in r.items() if k != "experiments"}
    if what == "long":
        r = {"metrics": r["metrics"], "summaries": [x.get("summary") for x in r["reports"]]}
    print(json.dumps(r, indent=1, default=str)[:20000])
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
