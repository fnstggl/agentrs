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
    """Digest of every source file of the `crucible` package (crucible.civ.longrun.code_key): a cached result is
    reused only by exactly the code that produced it (a change in any subsystem invalidates it)."""
    from .longrun import code_key
    return code_key()


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


# --------------------------------------------------------------------------------------------- SP-2.1b
# six seeds: with three, a paired 95% interval (t = 4.30) did not exclude zero although every seed's factual arm
# had the effect and no counterfactual arm had it (adopters 118/130/5 vs 0/0/0 at e77fef3: one seed's population
# found the service late); more seeds, not a weaker test
INVENTION_SEEDS = (51, 52, 53, 54, 55, 56)


def invention(seeds=INVENTION_SEEDS) -> dict:
    """SP-2.1b: the four kinds of consequence of an invention (crucible.civ.experiments.INVENTION_KINDS), each
    confirmed when one of its measures is confirmed by the counterfactual (no_invention arm) across seeds against the
    placebo reshuffling noise."""
    from . import experiments as X
    r = _safe("invention", "effect", seeds)
    eff = r["effects"].get("no_invention", {})
    kinds = {}
    for kind, ms in X.INVENTION_KINDS.items():
        kinds[kind] = {m: {k: eff[m][k] for k in ("effect_mean", "ci95", "reshuffle_noise_rms", "confirmed",
                                                   "factual", "intervention")} for m in ms if m in eff}
        kinds[kind]["confirmed"] = any(v.get("confirmed") for v in kinds[kind].values() if isinstance(v, dict))
    return {"invention_effect_kinds_confirmed": sum(1 for v in kinds.values() if v["confirmed"]), "kinds": kinds,
            "error": r.get("error"), "seeds": list(seeds), "scripted_inventor": True,
            "invariant_errors": r.get("invariant_errors")}


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
            per_seed = r.get("per_seed_chains", [])
            seeds_with = sum(1 for per in per_seed if any(c["actor_kind"] == "agent" and c.get(cls) for c in per))
            # stages after the root of the mined support chains that only a scripted actor produced
            scripted_stages = sum(1 for c in support for st in c["stages"]
                                  if st.get("scripted") and not st.get("is_root_stage"))
            unlabelled = sum(1 for c in support if not c.get("scripted"))
            chains_out.append({"experiment": exp, "arm": arm, "removed": what, "class": cls,
                               "confirmed_stages": stages, "n_confirmed_stages": len(stages),
                               "mined_in_reference_arm": len(support),
                               "max_mined_stages": max((c["n_stages"] for c in support), default=0),
                               "seeds_with_mined_chain": seeds_with, "seeds": len(per_seed),
                               "scripted_stages_after_root": scripted_stages, "unlabelled_scripted_roots": unlabelled,
                               "scripted_root": True,
                               "effects": {m: {k: v[k] for k in ("effect_mean", "ci95", "reshuffle_noise_rms",
                                                                 "confirmed")}
                                           for m, v in eff.items() if isinstance(v, dict)}})
    good = [c for c in chains_out if c["n_confirmed_stages"] >= 3 and c["mined_in_reference_arm"] > 0]
    return {"multi_stage_chains": len(good), "min_chain_stages": min((c["n_confirmed_stages"] for c in good),
                                                                     default=0),
            # SP-10.3b: the confirmation is across seeds (sign-consistent in every seed); replicated = also mined
            # from provenance in every seed's reference arm
            "chains_replicated_across_seeds": sum(1 for c in good if c["seeds"] and
                                                  c["seeds_with_mined_chain"] == c["seeds"]),
            "scripted_chain_stages": sum(c["scripted_stages_after_root"] for c in chains_out),
            "unlabelled_scripted_results": sum(c["unlabelled_scripted_roots"] for c in chains_out),
            "chains_with_agent_artifact": sum(1 for c in good if c["class"] == "agent_artifact"),
            "chains_with_social_activity": sum(1 for c in good if c["class"] == "social"),
            "chains_with_technical_activity": sum(1 for c in good if c["class"] == "technical"),
            "chains": chains_out, "experiments": raw}


# --------------------------------------------------------------------------------------------- SP-10.2
# coupling -> (profile whose world exercises its mechanism, the mechanism's own trace in the event log). The trace is
# what the coupling's mechanism records when it acts - never the downstream quantity it changes.
COUPLING_TRACE = {
    "science_process": ("balanced", lambda w, e: e.type == "links.rnd_check"),
    "compute_energy": ("balanced", lambda w, e: e.type == "links.energy"),
    "goods_supply": ("balanced", lambda w, e: e.type == "make.retail_sold" and _pop_buyer(e)),
    "goods_wom": ("balanced", lambda w, e: e.type == "goods.wom"),
    "org_coordination": ("balanced", lambda w, e: e.type == "links.coordination"),
    "freight_energy": ("balanced", lambda w, e: e.type == "links.load" and e.payload["coupling"] == "freight_energy"),
    "lab_energy": ("balanced", lambda w, e: e.type == "links.load" and e.payload["coupling"] == "lab_energy"),
    "energy_cost_pricing": ("balanced", lambda w, e: e.type in ("links.price_ref", "links.repriced")),
    "grid_curtailment": ("infrastructure_constrained", lambda w, e: e.type == "links.curtailed"),
    "datacenter_siting": ("infrastructure_constrained", lambda w, e: e.type == "links.dc"
                          and bool(e.payload.get("moved_from"))),
}
# a condition the experimenter sets at the end of the warm-up, before the fork, in every arm alike: the energy and
# siting couplings act only on a binding grid, so the datacenter regions' grids are cut to 10% of their capacity
COUPLING_CONDITIONS = {"infrastructure_constrained": ("datacenter region grids cut to 10% before the fork", 0.1)}
SCRIPTED_MODULES = ("shocks",)                  # scheduled exogenous narrative (never part of a civilisation)
COUPLING_RUN = {"seed": 31, "warm_days": 2, "arm_days": 4}


def _pop_buyer(e) -> bool:
    from ..substrate import ids
    return ids.index(str(e.payload.get("buyer", ""))) is not None


def _downstream(w, evs) -> dict:
    """What the couplings move, per arm (evaluator-side): grid energy, goods bought, compute prices, datacenter
    regions."""
    kwh = sum(float(e.payload["kwh"]) for e in evs if e.type in ("make.energy", "make.run"))
    return {"grid_kwh": round(kwh, 1), "goods_bought": sum(1 for e in evs if e.type == "make.retail_sold"),
            "compute_prices": sorted(int(r["price"]) for _, r in w.state.scan("providers")),
            "datacenter_regions": sorted(r["region"] for _, r in w.state.scan("civ_dc")),
            "backup_kwh": round(sum(float(e.payload["kwh"]) for e in evs if e.type == "links.backup" or
                                    (e.type == "links.load" and e.payload.get("backup"))), 1)}


def couplings_battery(seed: int | None = None, warm_days: int | None = None, arm_days: int | None = None) -> dict:
    """SP-10.2: every executable coupling of the catalogue is removed at a fork of a world whose mechanism exercises
    it, against a control fork with the same footprint (one operator command); removable = its trace is present in
    the control and absent after removal. Reusable = present in at least two structurally different profiles with
    only numeric parameters. Scripted consequence paths = scripted modules in any profile, one-off (non-periodic)
    jobs scheduled by the coupling engine, and scripted-module events in the battery's runs."""
    import torch
    torch.set_num_threads(1)
    from . import couplings as C, generator, profiles as P
    p = dict(COUPLING_RUN)
    p.update({k: v for k, v in (("seed", seed), ("warm_days", warm_days), ("arm_days", arm_days)) if v is not None})
    DAY = 86400

    def run():
        t = time.time()
        executable = sorted(c for c, rec in C.CATALOGUE.items() if not rec.pending)
        pending = sorted(c for c, rec in C.CATALOGUE.items() if rec.pending)
        by_prof: dict[str, list[str]] = {}
        for c in executable:
            by_prof.setdefault(COUPLING_TRACE[c][0], []).append(c)
        arms_out: dict[str, dict] = {}
        scripted = {"profiles_with_scripted_modules": [], "one_off_coupling_jobs": [], "scripted_events": 0}
        for prof, cids in sorted(by_prof.items()):
            w = generator.generate(prof, p["seed"])
            w.advance_time(int(p["warm_days"]) * DAY)
            if prof in COUPLING_CONDITIONS:
                from .experiments import cut_datacenter_grids
                cut_datacenter_grids(COUPLING_CONDITIONS[prof][1])(w)
            t0 = w.now
            for sid, j in w.state.scan("sched"):
                if j["proc"].startswith("civlinks.") and not j.get("period"):
                    scripted["one_off_coupling_jobs"].append(f"{prof}:{j['proc']}")
            res = {}
            for arm in ["control", *cids]:
                f = w.fork(f"sp102-{prof}-{arm}")
                s0 = f.seq
                if arm == "control":
                    f.system_call("admin.links_placebo", {"n": 1})
                else:
                    C.remove(f, arm)
                f.advance_time(t0 + int(p["arm_days"]) * DAY)
                evs = f.events_range(s0)
                res[arm] = {"trace": {c: sum(1 for e in evs if COUPLING_TRACE[c][1](f, e)) for c in cids},
                            "downstream": _downstream(f, evs), "events": len(evs),
                            "proc_failed": sum(1 for e in evs if e.type == "proc.failed")}
                scripted["scripted_events"] += sum(1 for e in evs if e.type.split(".", 1)[0] in SCRIPTED_MODULES)
                f.close()
            w.close()
            for c in cids:
                ctl, rem = res["control"]["trace"][c], res[c]["trace"][c]
                arms_out[c] = {"profile": prof, "trace_control": ctl, "trace_removed": rem,
                               "removable": ctl > 0 and rem == 0, "downstream_control": res["control"]["downstream"],
                               "downstream_removed": res[c]["downstream"], "proc_failed": res[c]["proc_failed"]}
        for name in sorted(P.PROFILES):
            mods = generator.recipe(name, p["seed"])["module_list"]
            bad = sorted(set(mods) & set(SCRIPTED_MODULES))
            if bad:
                scripted["profiles_with_scripted_modules"].append(f"{name}:{bad}")
        uses = {c: [n for n, pr in sorted(P.PROFILES.items()) if c in pr.couplings] for c in executable}
        generic = {c: all(isinstance(v, (int, float)) and not isinstance(v, bool)
                          for v in C.CATALOGUE[c].params.values()) for c in executable}
        structures = {c: len({P.structural_digest(P.PROFILES[n]) for n in uses[c]}) for c in executable}
        reusable = [c for c in executable if structures[c] >= 2 and generic[c]]
        removable = [c for c in executable if arms_out[c]["removable"]]
        n_scripted = len(scripted["profiles_with_scripted_modules"]) + len(scripted["one_off_coupling_jobs"]) + \
            int(scripted["scripted_events"])
        return {"params": p, "conditions": {k: v[0] for k, v in COUPLING_CONDITIONS.items()},
                "executable": executable, "pending": pending, "uses": uses, "reusable": reusable,
                "reusable_couplings": len(reusable), "removable": removable,
                "couplings_removable_fraction": round(len(removable) / max(1, len(executable)), 4),
                "scripted_consequence_paths": n_scripted, "scripted_detail": scripted, "arms": arms_out,
                "wall_s": round(time.time() - t, 1)}
    return cached("couplings", p, run)


# --------------------------------------------------------------------------------------------- SP-10.0
PERSIST_RUN = {"profile": "balanced", "seed": 41, "days": 8, "replications": 3, "free_reruns": 2, "fork_day": 4}
# entity kind -> state tables holding entities of that kind (Part 10.4: participants, organisations, relationships,
# products, technologies, institutions; communities and content as well)
ENTITY_TABLES = {
    "organisations": ("orgs", "providers", "facilities"),
    "relationships": ("forum_follows", "members", "social_edges", "follows"),
    "products": ("products", "plans", "retail", "subscriptions"),
    "technologies": ("designs", "machines", "deployments", "repos", "capabilities", "cap_decls", "process_notes",
                     "sci_papers"),
    "institutions": ("constitutions", "inst_charters", "civ_links"),
    "communities": ("communities", "forum_boards"),
    "content": ("content", "forum_posts"),
}


def _replicate_child(profile: str, seed: int, until: int) -> dict:
    """A free rerun: a fresh process builds the same spec and seed and runs it again, recomputing every neural
    inference (reported, not counted: see `_regenerate_child`)."""
    import subprocess
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    code = ("import json,sys,torch; torch.set_num_threads(1); import crucible; crucible.load_all_modules(); "
            "from crucible.civ import generator as g; "
            f"w=g.generate({profile!r},{int(seed)}); w.advance_time({int(until)}); "
            "print(json.dumps({'hash':w.last_hash,'digest':w.digest(),'seq':w.seq}))")
    pr = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, timeout=3600)
    if pr.returncode != 0:
        return {"ok": False, "error": pr.stderr.decode(errors="replace")[-800:]}
    return {"ok": True, **json.loads(pr.stdout.decode().strip().splitlines()[-1])}


def _regenerate_child(store_path: str, world_id: str, to_seq: int) -> dict:
    """An independent replication under contract 1.7 (crucible.substrate.inference): a fresh process reads only the
    spec and the recorded log of the run from its store and re-executes it from genesis through the handlers
    (`World.regenerate`): every module's handlers and processes, every population decision and every random draw
    run again; only the outcomes that are recorded because they are not bit-reproducible (neural inference, measured
    sandbox CPU time: kernel `ext` events) are handed back from the record. It must reach the run's event hash and
    state digest."""
    import subprocess
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    code = ("import json,sys,time,torch; torch.set_num_threads(1); import crucible; crucible.load_all_modules(); "
            "from crucible.core.kernel import World; from crucible.storage.sqlite import SQLiteStore; "
            "from crucible.civ import history as H; "
            f"st=SQLiteStore({store_path!r}); sp=H._spec(st,{world_id!r}); "
            f"evs=st.read_events({world_id!r},from_seq=0,to_seq={int(to_seq)}); t=time.time(); "
            "g=World.regenerate(sp,evs,world_id='sp100-regen',store=st); "
            "print(json.dumps({'hash':g.last_hash,'digest':g.digest(),'seq':g.seq,'events_read':len(evs),"
            "'wall_s':round(time.time()-t,1)}))")
    pr = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, timeout=3600)
    if pr.returncode != 0:
        return {"ok": False, "error": pr.stderr.decode(errors="replace")[-1500:]}
    return {"ok": True, **json.loads(pr.stdout.decode().strip().splitlines()[-1])}


def _census(w) -> dict:
    """{kind: {entity id: record digest}} (participants: per-individual digest of the population's own columns)."""
    from ..core.canonical import digest
    from ..substrate import columns
    st = w.state
    out: dict[str, dict] = {}
    for kind, tables in ENTITY_TABLES.items():
        d = {}
        for t in tables:
            if t in st.tables:
                for k, r in st.scan(t):
                    d[f"{t}:{k}"] = digest(dict(r), size=8)
        out[kind] = d
    if "pop" in columns.tables(w):
        cols = [c for c in ("n_sessions", "satisfaction", "spent_cents", "last_activity") if
                c in columns.tables(w)["pop"]["columns"]]
        arr = [columns.col(w, "pop", c) for c in cols]
        n = columns.rows(w, "pop")
        out["participants"] = {f"p{i}": digest([float(a[i]) for a in arr], size=8) for i in range(n)}
    # an organisation also evolves through its money: the balances of its accounts
    bank = w.modules.get("bank")
    if bank is not None:
        for aid, a in st.scan("accounts"):
            key = f"orgs:{a['owner']}"
            if key in out["organisations"]:
                out["organisations"][key] += f"|{bank.balance(st, aid)}"
    return out


def persistence(profile: str | None = None, seed: int | None = None, days: int | None = None) -> dict:
    """SP-10.0 on one store-backed civilisation: forks (an unchanged fork reproduces its parent, an intervened one
    diverges, the parent is untouched), the historical-query kinds of crucible.civ.history (each verified),
    independent replications (fresh processes that rebuild the same spec and seed and must reach the same event hash
    and state digest), entity kinds that persist and evolve, and the standalone (Mode A/B) worlds and Tier 3
    profiles that training can draw from."""
    import shutil
    import torch
    torch.set_num_threads(1)
    from ..storage.sqlite import SQLiteStore
    from . import couplings as C, generator, history, profiles as P
    p = dict(PERSIST_RUN)
    p.update({k: v for k, v in (("profile", profile), ("seed", seed), ("days", days)) if v is not None})
    DAY = 86400

    def run():
        t = time.time()
        wd = tempfile.mkdtemp(prefix="civ-sp100-")
        try:
            store = SQLiteStore(os.path.join(wd, "civ.sqlite"))
            w = generator.generate(p["profile"], p["seed"], store=store, world_id="sp100")
            w.snapshot()
            w.advance_time(DAY)
            day1 = {"hash": w.last_hash, "digest": w.digest(), "seq": w.seq}
            w.snapshot()
            c1 = _census(w)
            w.advance_time(int(p["fork_day"]) * DAY)
            w.snapshot()
            # forks: an exact copy at the fork, independent of its parent, diverging under an intervention; an
            # unchanged fork reproduces its parent's continuation unless a recorded non-reproducible outcome
            # (inference numerics) differs first - the fork itself must never be the origin of a divergence
            before, h_before, s_fork = w.digest(), w.last_hash, w.seq
            f_same, f_int = w.fork("sp100-same"), w.fork("sp100-int")
            exact_copy = all(f.digest() == before and f.last_hash == h_before and f.seq == s_fork
                             for f in (f_same, f_int))
            C.remove(f_int, "goods_wom")
            t_f = w.now + DAY
            f_same.advance_time(t_f)
            f_int.advance_time(t_f)
            parent_untouched = w.digest() == before and w.last_hash == h_before
            w.advance_time(t_f)
            same_equal = f_same.digest() == w.digest() and f_same.last_hash == w.last_hash
            from .determinism import first_divergence
            div = first_divergence(list(w.events_range(s_fork)), list(f_same.events_range(s_fork)))
            forks = {"parent_untouched": parent_untouched, "fork_exact_copy": exact_copy,
                     "unchanged_fork_equals_parent": same_equal,
                     "unchanged_fork_divergence": div,
                     "unchanged_fork_consistent": same_equal or div["origin"] == "recorded_outcome",
                     "intervened_fork_diverges": f_int.digest() != f_same.digest()}
            forks["fork_divergence_supported"] = bool(forks["parent_untouched"] and exact_copy and
                                                      forks["unchanged_fork_consistent"] and
                                                      forks["intervened_fork_diverges"])
            f_same.close()
            f_int.close()
            w.advance_time(max(int(p["days"]), int(p["fork_day"]) + 2) * DAY)
            w.snapshot()
            c2 = _census(w)
            hist = history.verify(w)
            kinds = {}
            for kind in sorted(set(c1) | set(c2)):
                a, b = c1.get(kind, {}), c2.get(kind, {})
                persisted = len(set(a) & set(b))
                changed = sum(1 for k in set(a) & set(b) if a[k] != b[k])
                new = len(set(b) - set(a))
                kinds[kind] = {"at_day_1": len(a), "at_end": len(b), "persisted": persisted, "changed": changed,
                               "new": new, "persists_and_evolves": persisted > 0 and (changed + new) > 0}
            w.close()
            # independent replications: fresh processes regenerate the first simulated day from the record (counted);
            # free reruns that recompute every inference are run as well and reported (GPU numerics of batched
            # inference are not bit-reproducible, which is why contract 1.7 records those outcomes)
            reps = [_regenerate_child(store.path, "sp100", day1["seq"]) for _ in range(int(p["replications"]))]
            rep_ok = [r for r in reps if r.get("ok") and r["hash"] == day1["hash"] and r["digest"] == day1["digest"]
                      and r["seq"] == day1["seq"]]
            free = [_replicate_child(p["profile"], p["seed"], DAY) for _ in range(int(p["free_reruns"]))]
            free_ok = [r for r in free if r.get("ok") and r["hash"] == day1["hash"] and r["digest"] == day1["digest"]]
            from ..substrate import modes as M
            M.discover()
            standalone = {}
            for mode in ("A", "B"):
                n = 0
                for prov in M.providers(mode):
                    try:
                        n += len(prov.names())
                    except Exception:  # noqa: BLE001 - a provider that cannot list counts as none
                        pass
                standalone[mode] = n
            return {"params": p, "forks": forks, "fork_divergence_supported": forks["fork_divergence_supported"],
                    "history": hist, "historical_query_kinds": hist["historical_query_kinds"],
                    "replications": reps, "independent_replications": len(rep_ok), "day1": day1,
                    "free_reruns": free, "free_reruns_exact": len(free_ok),
                    "entity_kinds": kinds,
                    "entity_kinds_persisting_and_evolving": sum(1 for v in kinds.values()
                                                                if v["persists_and_evolves"]),
                    "standalone_environments": standalone, "tier3_profiles": len(P.PROFILES),
                    "wall_s": round(time.time() - t, 1)}
        finally:
            shutil.rmtree(wd, ignore_errors=True)
    return cached("persistence", p, run)


# --------------------------------------------------------------------------------------------- SP-19.3
def vertical_slices() -> dict:
    """SP-19.3: the seven Phase 3 vertical slices (crucible.civ.slices.run_all), executed and recorded."""
    from . import slices
    return cached("slices", {"civ": slices.CIV_RUN, "learner": slices.LEARNER_RUN,
                             "rapid": list(slices.RAPID_FAMILIES)}, slices.run_all)


# --------------------------------------------------------------------------------------------- SP-17.1
def redemonstration() -> dict:
    """SP-17.1: crucible.civ.redemo.run_all (world gates measured with their own code on civilisations)."""
    from . import redemo
    return cached("redemo", {"creation": redemo.CREATION, "autonomous": redemo.AUTONOMOUS, "d": redemo.D_RUN,
                             "g": redemo.G_RUN, "h": redemo.H_RUN},
                  lambda: redemo.run_all(workers=workers()))


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
        reps = longrun.suite(horizons, seeds_per_horizon, workers=workers(), max_long=1,
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
    m["snapshot_resume_exact_checks"] = sum(s.get("resume_exact_checks", 0) for s in S)
    m["free_resume_equal"] = sum(s.get("free_resume_equal", 0) for s in S)
    m["free_resume_divergence_origins"] = ",".join(sorted({o for s in S for o in s.get("free_resume_divergence_origins")
                                                           or []})) or "none"
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
        seeds = CHAIN_SEEDS if kind == "chain" else INVENTION_SEEDS if argv[1] == "invention" else EFFECT_SEEDS
        r = _safe(argv[1], kind, seeds)
        print(json.dumps({"experiment": argv[1], "confirmed": r.get("confirmed"), "error": r.get("error"),
                          "effects": r.get("effects")}, indent=1, default=str)[:20000])
        return 0
    fn = {"integration": integration, "effects": effects, "chains": chain_results, "structure": structure,
          "long": long_horizon, "couplings": couplings_battery, "persistence": persistence,
          "invention": invention, "slices": vertical_slices, "redemo": redemonstration}[what]
    r = fn()
    if what in ("effects", "chains"):
        r = {k: v for k, v in r.items() if k != "experiments"}
    if what == "long":
        r = {"metrics": r["metrics"], "summaries": [x.get("summary") for x in r["reports"]]}
    print(json.dumps(r, indent=1, default=str)[:20000])
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
