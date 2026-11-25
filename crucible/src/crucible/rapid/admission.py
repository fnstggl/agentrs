"""World admission (MASTER_SPEC Part 7.4): ten checks every family must pass on generated worlds before it counts.

    consistency           same seed and actions give the same observations; observations are valid section-11
                          Observations; states stay finite; malformed actions are refused; the config is not mutated
    conservation          every engine's accounting identities (jobs, stock, flow, budgets, money) hold every tick,
                          and cash = endowment + revenue - cost
    controllability       outcomes under different policies differ by more than the world's own noise
    achievable            on the same world, the private oracle does clearly better (by more than 1%) than leaving
                          it alone, and at least as well as every trivial strategy
    informative           under a fixed policy (waiting, or seeded random actions), time-averaged observations vary
                          between instances of the family (different hidden parameters) far more than between noise
                          draws of one instance (the F statistic of a one-way analysis of variance); for hidden-state
                          machines, alternatively, the hidden state is predictable from the current readings
    identifiability       the family's experimental question is answered correctly from public actions and
                          observations (`identify`)
    no_dominant_strategy  no constant canonical action and no random policy gets close to the oracle
    difficulty            a scripted public-information reference reaches a meaningful but not saturated share of the
                          oracle's improvement; every difficulty axis the family declares changes the world
    no_leakage            no hidden parameter appears in anything a participant receives, no observation key names
                          evaluator data, and world code cannot import evaluator code (static check)
    distinct              the fingerprint differs from every other registered v5 family and every Crucible 4.0
                          mechanism family

Oracles, references, intervention tests and identification run only here, on copies of worlds, evaluator side.
Checks that are statistical (controllability, informative, identifiability, no_dominant_strategy, difficulty) must
pass on at least two thirds of the seeds; the others on every seed. `admit(family, seeds)` returns the report.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time

import numpy as np

from ..core.canonical import digest
from . import observation as O
from . import policies as P
from .env import Env
from .families import DIFFICULTY_AXES, FAMILIES, build_config, make

CHECKS = ("consistency", "conservation", "controllability", "achievable", "informative", "identifiability",
          "no_dominant_strategy", "difficulty", "no_leakage", "distinct")
SOFT = ("controllability", "informative", "identifiability", "no_dominant_strategy", "difficulty")
DOMINANCE_MAX = 0.75          # a trivial strategy may reach at most this share of the oracle's improvement
REFERENCE_RANGE = (0.05, 0.98)
EVALUATOR_MODULES = ("crucible.rapid.oracles", "crucible.rapid.admission", "crucible.rapid.claims",
                     "crucible.rapid.identify", "crucible.rapid.references", "crucible.rapid.fingerprints",
                     "crucible.rapid.primitive_checks", "crucible.rapid.report", "crucible.mechanisms.evaluator",
                     "crucible.mechanisms.validity", "crucible.rapid.evaluator_h2")
WORLD_MODULES = ("crucible.rapid.env", "crucible.rapid.engines", "crucible.rapid.families", "crucible.rapid.actions",
                 "crucible.rapid.observation", "crucible.rapid.mechview", "crucible.rapid.batch",
                 "crucible.rapid.program", "crucible.rapid.generator", "crucible.rapid.primitives")

# config keys that hold hidden answers, by engine (numbers under them must never reach a participant)
SECRET_KEYS = {
    "primitive_graph": ("params", "ports", "bias", "natural", "money", "heldout_values"),
    "discrete_event": ("mu", "arrival_rate", "eta", "shape", "lead_p", "demand", "season_amp", "season_period",
                       "season_phase", "shelf_life", "repair_time", "service_time"),
    "graph_dynamical": ("c1", "c2", "toll", "demand_mean", "demand_sd", "theta", "cap", "base", "hazard",
                        "pair_demand", "shock", "rewire_at"),
    "state_machine": ("T", "slope", "base", "loss", "emit", "lever_cost"),
    "search": ("bumps", "weights", "app", "features_true"),
}


# ====================================================================== helpers
def _fork(env):
    from .oracles import fork
    return fork(env)


def _value(env, policy, rng_seed=0, others=None) -> float:
    return float(P.run(_fork(env), policy, np.random.default_rng(rng_seed), agents=others)["value"])


def _replicate(env, r: int) -> Env:
    return Env(env.family, env.seed, env._config, env.difficulty, world_seed=env.world_seed + 104729 * (r + 1))


def _floats(x, out: set, keyok: bool):
    if isinstance(x, dict):
        for k, v in x.items():
            _floats(v, out, keyok)
    elif isinstance(x, list):
        for v in x:
            _floats(v, out, keyok)
    elif isinstance(x, float) and keyok and math.isfinite(x) and x != round(x) and len(repr(x)) >= 7:
        out.add(repr(x))


def secrets(env) -> set[str]:
    """Hidden numbers of a world (repr strings), minus any number that is part of the public spec or schemas."""
    cfg = env._config
    keys = SECRET_KEYS.get(cfg["engine"]) or tuple(getattr(env._engine, "SECRET_KEYS", ()))
    out: set[str] = set()
    for k in keys:
        if k in cfg:
            _floats(cfg[k], out, True)
    if cfg["engine"] == "primitive_graph":
        for nd in cfg["nodes"]:
            for k in ("params", "ports", "bias", "natural"):
                if k in nd:
                    _floats(nd[k], out, True)
        for g in cfg.get("gauges", []):
            for item in g.get("chain", []):
                _floats(item.get("params", {}), out, True)
    pub: set[str] = set()
    _floats(json.loads(json.dumps(env.spec, default=str)), pub, True)
    for a in env.agents():
        _floats(env.action_space(a).public(), pub, True)
    return out - pub


def leak_scan(texts: list[str], secret: set[str]) -> list[str]:
    found = []
    for s in secret:
        for t in texts:
            if s in t:
                found.append(s)
                break
    return found


def static_isolation(root: str | None = None) -> dict:
    """Import every world-side rapid module in a fresh interpreter and list evaluator modules that came with them."""
    import os
    root = root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    code = ("import sys, json, importlib; sys.path.insert(0, 'src'); "
            f"[importlib.import_module(m) for m in {list(WORLD_MODULES)!r}]; "
            "import crucible.rapid as R; R.make('net_routing', 0).step(None); "
            f"print(json.dumps(sorted(m for m in sys.modules if m in {list(EVALUATOR_MODULES)!r})))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root, timeout=120)
    if r.returncode != 0:
        return {"ok": False, "error": r.stderr[-800:], "imported": None}
    bad = json.loads(r.stdout.strip().splitlines()[-1])
    return {"ok": not bad, "imported": bad, "world_modules": list(WORLD_MODULES)}


_LEGACY_FP: dict | None = None


def legacy_fingerprints() -> dict:
    global _LEGACY_FP
    if _LEGACY_FP is None:
        from ..mechanisms.base import mech_families
        from ..mechanisms.sim import ensure_engines
        from .fingerprints import legacy_fingerprint
        ensure_engines()
        _LEGACY_FP = {f: legacy_fingerprint(f) for f in mech_families()}
    return _LEGACY_FP


def distinctness(family: str) -> dict:
    from .fingerprints import distance, family_fingerprint, nearest
    fps = {n: family_fingerprint(n) for n in FAMILIES}
    own = fps[family]
    v5 = nearest(family, fps)
    leg = legacy_fingerprints()
    d_leg = {k: distance(own, fp) for k, fp in leg.items()}
    lk = min(d_leg, key=d_leg.get) if d_leg else None
    ok = (v5["distance"] is None or v5["distance"] > 0) and (lk is None or d_leg[lk] > 0)
    return {"ok": bool(ok), "nearest_v5": v5["nearest"], "distance_v5": v5["distance"],
            "differs_in": v5["differs_in"], "nearest_4_0": lk, "distance_4_0": d_leg.get(lk) if lk else None}


# ====================================================================== per-seed checks
def _consistency(family, seed, difficulty) -> dict:
    a, b = make(family, seed, **difficulty), make(family, seed, **difficulty)
    cfg_before = digest(a._config)
    ra, rb = np.random.default_rng(7), np.random.default_rng(7)
    obs_errs, state_errs, trace_a, trace_b = [], [], [], []
    others_a = {x: P.random_policy for x in a.agents()[1:]}
    for env, rng, trace in ((a, ra, trace_a), (b, rb, trace_b)):
        obs = {x: env.observe(x) for x in env.agents()}
        trace.append(digest(obs))
        while not env.done:
            acts = {}
            for ag in env.agents():
                x = P.random_policy(obs[ag], env.action_space(ag), rng)
                if x is not None:
                    acts[ag] = x
            r = env.step(acts if len(env.agents()) > 1 else acts.get(env.agents()[0]))
            obs = r.get("observations") or {env.agents()[0]: r["observation"]}
            for o in obs.values():
                e = O.check(o)
                if e and env is a:
                    obs_errs += e[:2]
            if env is a:
                state_errs += env._engine.check_state(env._state)[:2]
            trace.append(digest(obs))
    del others_a
    # malformed actions are refused (and the tick still passes)
    c = make(family, seed, **difficulty)
    t0 = c.t
    bad = [{"kind": "no_such_kind"}, {"kind": sorted(c.action_space().kinds)[0], "unexpected": 1}, "not an object"]
    if "levers" in c.action_space().kinds:
        props = c.action_space().kinds["levers"]["properties"]["values"]["properties"]
        n = sorted(props)[0]
        bad.append({"kind": "levers", "values": {n: props[n]["maximum"] + 1e6}})
    refused = [not c.step(x)["result"]["ok"] for x in bad]
    return {"ok": trace_a == trace_b and not obs_errs and not state_errs and all(refused) and c.t == t0 + len(bad)
            and digest(a._config) == cfg_before, "deterministic": trace_a == trace_b,
            "observation_errors": obs_errs[:5], "state_errors": state_errs[:5], "malformed_refused": refused,
            "config_unchanged": digest(a._config) == cfg_before, "ticks": len(trace_a) - 1}


def _conservation(env_factory) -> dict:
    errs, ledger = [], []
    for k, pol in enumerate((P.random_policy, P.wait)):
        env = env_factory()
        env._config = dict(env._config, _audit=True)
        P.run(env, pol, np.random.default_rng(11 + k), agents={x: P.random_policy for x in env.agents()[1:]})
        errs += getattr(env, "_audit", [])
        for a in env.agents():
            exp = float(env._config.get("endowment", 0)) + env._ledger["revenue"][a] - env._ledger["cost"][a]
            if abs(env._ledger["cash"][a] - exp) > 1e-6 * max(1.0, abs(exp)):
                ledger.append(a)
            if not all(math.isfinite(x) for x in (env._ledger["revenue"][a], env._ledger["cost"][a])):
                ledger.append(a)
    return {"ok": not errs and not ledger, "violations": errs[:5], "ledger_mismatch": ledger}


def _flat_numbers(obs: dict) -> list[float]:
    out: list[float] = []

    def walk(x):
        if isinstance(x, bool):
            out.append(float(x))
        elif isinstance(x, (int, float)):
            if math.isfinite(x):
                out.append(float(x))
        elif isinstance(x, dict):
            for k in x:
                walk(x[k])
        elif isinstance(x, list):
            for v in x:
                walk(v)
    for f in ("numbers", "arrays", "entities", "resources", "last_result"):
        walk(obs.get(f))
    return out


def _stable_numbers(obs: dict) -> list[float]:
    out: list[float] = []
    for f in ("numbers", "resources"):
        out += [float(v) for v in obs[f].values() if isinstance(v, (int, float)) and math.isfinite(v)]
    for v in obs["arrays"].values():
        out += [float(x) for x in v if isinstance(x, (int, float)) and math.isfinite(x)]
    for e in obs["entities"]:
        out += [float(v) for v in e.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        out += [float(v) for v in e.values() if isinstance(v, bool)]
    return out


def _result_leaves(x, path="r", out=None) -> dict:
    out = {} if out is None else out
    if isinstance(x, dict):
        for k, v in x.items():
            _result_leaves(v, f"{path}.{k}", out)
    elif isinstance(x, list):
        for i, v in enumerate(x[:16]):
            _result_leaves(v, f"{path}[{i}]", out)
    elif isinstance(x, bool):
        out[path] = float(x)
    elif isinstance(x, (int, float)) and math.isfinite(x):
        out[path] = float(x)
    return out


def _features(env, policy: str, rng_seed=3) -> tuple[np.ndarray, list, list]:
    """Time-averaged numeric observations (by position) under a fixed policy; also, per tick, the observation
    features (readings plus the numbers in the participant's own action results) and the hidden state of
    engines that expose one (`latent`, evaluator side)."""
    rng = np.random.default_rng(rng_seed)
    rows, ticks, hidden = [], [], []
    obs = {a: env.observe(a) for a in env.agents()}
    latent = getattr(env._engine, "latent", None)
    while not env.done:
        if policy == "wait":
            acts = {}
        else:
            acts = {a: P.random_policy(obs[a], env.action_space(a), rng) for a in env.agents()}
        r = env.step(acts if len(env.agents()) > 1 else acts.get(env.agents()[0]))
        obs = r.get("observations") or {env.agents()[0]: r["observation"]}
        rows.append([x for a in env.agents() for x in _stable_numbers(obs[a])])
        f = {}
        for a in env.agents():
            f.update({f"{a}.n{i}": v for i, v in enumerate(_stable_numbers(obs[a]))})
            f.update({f"{a}.{k}": v for k, v in _result_leaves(obs[a]["last_result"]).items()})
        ticks.append(f)
        if latent is not None:
            hidden.append(np.asarray(latent(env._state), dtype=float))
    m = min(len(x) for x in rows) if rows else 0
    return (np.asarray([x[:m] for x in rows]).mean(axis=0) if m else np.zeros(0)), ticks, hidden


def _f_ratio(groups) -> float:
    m = min(len(f) for g in groups for f in g)
    if m == 0:
        return 0.0
    X = np.asarray([[f[:m] for f in g] for g in groups])           # (K, R, m)
    R = X.shape[1]
    within = X.var(axis=1, ddof=1).mean(axis=0) / R + 1e-12
    F = X.mean(axis=1).var(axis=0, ddof=1) / within
    keep = np.isfinite(F)
    return float(F[keep].max()) if keep.any() else 0.0


def _informative(family, seed, difficulty, K: int = 4, R: int = 3) -> dict:
    out = {}
    for pol in ("wait", "random"):
        groups = []
        for k in range(K):
            base = make(family, seed + 1000 * k, **difficulty)
            groups.append([_features(_replicate(base, r), pol)[0] for r in range(R)])
        out[pol] = _f_ratio(groups)
    best = max(out.values())
    res = {"between_within_ratio": out, "instances": K, "noise_draws": R}
    ok = best >= 10.0
    env = make(family, seed, **difficulty)
    if hasattr(env._engine, "latent"):
        # is the hidden state predictable from what is observed? (out-of-sample R^2 of a ridge read-out, first
        # half of the ticks to fit, second half to test, pooled over the two fixed policies and two worlds)
        F_, Y = [], []
        for pol in ("wait", "random"):
            for r in range(2):
                _, ticks, hid = _features(_replicate(env, r), pol)
                F_ += ticks
                Y += hid
        keys = sorted({k for f in F_ for k in f})
        X = np.asarray([[f.get(k, 0.0) for k in keys] for f in F_])
        Y = np.asarray(Y)
        r2s = []
        n = len(X)
        idx = np.arange(n)
        tr, te = idx % 2 == 0, idx % 2 == 1
        for j in range(Y.shape[1]):
            y = Y[:, j]
            if y[te].var() <= 0:
                continue
            A = np.c_[np.ones(tr.sum()), X[tr]]
            beta = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ y[tr])
            pred = np.c_[np.ones(te.sum()), X[te]] @ beta
            r2s.append(float(1 - ((y[te] - pred) ** 2).mean() / y[te].var()))
        if r2s:
            res["hidden_state_r2"] = max(r2s)
            ok = ok or max(r2s) >= 0.3
    return dict(res, ok=bool(ok))


def _axes_effective(family, seed) -> dict:
    fam = FAMILIES[family]
    base = digest(build_config(family, seed))
    out = {}
    for ax, v in fam.axes.items():
        nv = v * 1.5 if isinstance(v, float) else (int(v) + max(1, int(v) // 2) if v else 1)
        if not v:
            nv = 1.0 if isinstance(v, float) else 1
        try:
            out[ax] = digest(build_config(family, seed, **{ax: nv})) != base
        except Exception as e:  # noqa: BLE001 - an axis that crashes the builder is not effective
            out[ax] = f"error: {type(e).__name__}: {e}"
    return out


def seed_report(family: str, seed: int, difficulty: dict | None = None) -> dict:
    from .identify import run as identify_run
    from .oracles import oracle
    from .references import play
    difficulty = dict(difficulty or {})
    t0 = time.process_time()
    env = make(family, seed, **difficulty)
    rep: dict = {"seed": seed, "checks": {}}
    ck = rep["checks"]
    ck["consistency"] = _consistency(family, seed, difficulty)
    ck["conservation"] = _conservation(lambda: make(family, seed, **difficulty))
    others = {a: P.random_policy for a in env.agents()[1:]}
    v_def = _value(env, P.wait)
    triv = {lab: _value(env, P.constant(a)) for lab, a in P.canonical_actions(env.action_space())}
    triv["random"] = _value(env, P.random_policy, 1, others)
    triv["wait"] = v_def
    orc = oracle(_fork(env))
    v_or = float(orc["value"])
    v_ref = float(play(make(family, seed, **difficulty))["value"])
    reps = [_value(_replicate(env, r), P.random_policy, 1, others) for r in range(3)] + \
        [_value(_replicate(env, r), P.wait) for r in range(3)]
    noise_sd = float(np.std(reps[:3]) + np.std(reps[3:])) / 2.0
    vals = list(triv.values()) + [v_or, v_ref]
    spread = float(max(vals) - min(vals))
    gap = v_or - v_def
    ck["controllability"] = {"ok": spread > 3.0 * noise_sd + 1e-9 and spread > 1e-6, "outcome_spread": spread,
                             "noise_sd": noise_sd}
    margin = 0.01 * (abs(v_def) + abs(v_or)) / 2 + 1e-6
    ck["achievable"] = {"ok": gap > margin and v_or >= max(triv.values()), "oracle": v_or, "default": v_def,
                        "margin": margin, "how": orc.get("how"), "objective": env._config["objective"]["type"]}
    ck["informative"] = _informative(family, seed, difficulty)
    ident = identify_run(family, seed, **difficulty)
    ck["identifiability"] = {"ok": bool(ident.get("ok")), **{k: ident.get(k) for k in ("question", "estimate",
                                                                                         "truth")}}
    shares = {k: (v - v_def) / gap if gap > 0 else float("inf") for k, v in triv.items()}
    worst = max(shares, key=shares.get)
    ck["no_dominant_strategy"] = {"ok": gap > 0 and shares[worst] <= DOMINANCE_MAX, "best_trivial": worst,
                                  "best_trivial_share": float(shares[worst]) if gap > 0 else None,
                                  "threshold": DOMINANCE_MAX}
    s_ref = (v_ref - v_def) / gap if gap > 0 else float("nan")
    axes = _axes_effective(family, seed)
    ck["difficulty"] = {"ok": gap > 0 and REFERENCE_RANGE[0] <= s_ref <= REFERENCE_RANGE[1]
                        and all(v is True for v in axes.values()), "reference_share": float(s_ref)
                        if s_ref == s_ref else None, "range": list(REFERENCE_RANGE), "axes_effective": axes}
    # leakage: everything a participant received in the reference and random runs
    sec = secrets(env)
    texts = []
    for e in (make(family, seed, **difficulty), make(family, seed, **difficulty)):
        if e is not None:
            rng = np.random.default_rng(5)
            obs = {a: e.observe(a) for a in e.agents()}
            texts.append(json.dumps(obs, default=str))
            while not e.done:
                acts = {a: P.random_policy(obs[a], e.action_space(a), rng) for a in e.agents()}
                r = e.step(acts if len(e.agents()) > 1 else acts[e.agents()[0]])
                obs = r.get("observations") or {e.agents()[0]: r["observation"]}
                texts.append(json.dumps(r, default=str))
            texts.append(json.dumps(e.spec, default=str))
            texts.append(json.dumps([e.action_space(a).public() for a in e.agents()], default=str))
    found = leak_scan(texts, sec)
    key_errs = [x for t in texts[:1] for x in O.check(json.loads(t)[env.agents()[0]])] if texts else []
    ck["no_leakage"] = {"ok": not found and not key_errs, "secrets_checked": len(sec), "leaked": found[:5],
                        "key_errors": key_errs[:3]}
    rep["values"] = {"default": v_def, "oracle": v_or, "reference": v_ref, "trivial": triv}
    rep["cpu_s"] = time.process_time() - t0
    return rep


def admit(family: str, seeds=(0, 1, 2), **difficulty) -> dict:
    """The Part 7.4 checks on `seeds`; evaluator side. Returns {"family", "admitted", "feasible", "checks",
    "seeds", "cpu_s"}."""
    t0 = time.process_time()
    reps = [seed_report(family, int(s), difficulty) for s in seeds]
    dist = distinctness(family)
    checks = {}
    for name in CHECKS:
        if name == "distinct":
            checks[name] = dict(dist, required=1, seeds_passed=None)
            continue
        oks = [bool(r["checks"][name]["ok"]) for r in reps]
        need = math.ceil(2 * len(reps) / 3) if name in SOFT else len(reps)
        checks[name] = {"ok": sum(oks) >= need, "seeds_passed": sum(oks), "required": need,
                        "per_seed": [r["checks"][name] for r in reps]}
    feasible = all(checks[k]["ok"] for k in ("consistency", "conservation", "achievable"))
    return {"family": family, "engine": FAMILIES[family].engine, "split": FAMILIES[family].split,
            "category": FAMILIES[family].category, "admitted": all(c["ok"] for c in checks.values()),
            "feasible": feasible, "checks": checks, "seeds": list(seeds),
            "values": [r["values"] for r in reps], "cpu_s": time.process_time() - t0,
            "difficulty_axes": sorted(DIFFICULTY_AXES)}


def admit_all(families: list[str] | None = None, seeds=(0, 1, 2)) -> dict[str, dict]:
    from .families import families as all_families
    return {f: admit(f, seeds) for f in (families or all_families())}


def summary(reports: dict[str, dict]) -> dict:
    """Counts the gates need, measured from admission reports and the registry."""
    adm = {f: r for f, r in reports.items() if r["admitted"]}
    return {"families": len(reports), "admitted": len(adm),
            "admitted_train": sum(1 for r in adm.values() if r["split"] == "train"),
            "admitted_eval": sum(1 for r in adm.values() if r["split"] in ("val", "test")),
            "categories_admitted": sorted({r["category"] for r in adm.values()}),
            "engines_admitted": sorted({r["engine"] for r in adm.values()}),
            "feasibility_validated": sum(1 for r in reports.values() if r["feasible"]),
            "feasibility_failures_admitted": sum(1 for r in adm.values() if not r["feasible"]),
            "failed_checks": {f: [k for k, c in r["checks"].items() if not c["ok"]] for f, r in reports.items()
                              if not r["admitted"]}}
