"""World admission (MASTER_SPEC Part 7.4): ten checks every family must pass on generated worlds before it counts.

    consistency           same seed and actions give the same observations; observations are valid section-11
                          Observations; states stay finite; malformed actions are refused; the config is not mutated
    conservation          every engine's accounting identities (jobs, stock, flow, budgets, money) hold every tick,
                          and cash = endowment + revenue - cost
    controllability       outcomes under different policies differ by more than the world's own noise
    achievable            the private oracle reaches a clearly better outcome than leaving the world alone
    informative           observations under one fixed policy differ more between two instances of the family
                          (different hidden structure) than between two noise draws of one instance
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
                     "crucible.rapid.identify", "crucible.rapid.references", "crucible.rapid.fingerprint",
                     "crucible.rapid.primitive_checks", "crucible.mechanisms.evaluator", "crucible.mechanisms.validity")
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
    keys = SECRET_KEYS.get(cfg["engine"], ())
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
        from .fingerprint import legacy_fingerprint
        ensure_engines()
        _LEGACY_FP = {f: legacy_fingerprint(f) for f in mech_families()}
    return _LEGACY_FP


def distinctness(family: str) -> dict:
    from .fingerprint import distance, family_fingerprint, nearest
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


def _stream(env, rng_seed=3) -> list[list[float]]:
    rng = np.random.default_rng(rng_seed)
    out = []
    obs = env.observe()
    while not env.done:
        x = P.random_policy(obs, env.action_space(), rng)
        if len(env.agents()) > 1:
            r = env.step({a: P.random_policy(env.observe(a), env.action_space(a), rng) for a in env.agents()})
        else:
            r = env.step(x)
        obs = r["observation"]
        out.append(_flat_numbers(obs))
    return out


def _stream_distance(a, b) -> float:
    tot, n = 0.0, 0
    for x, y in zip(a, b):
        m = min(len(x), len(y))
        if len(x) != len(y):
            tot += 1.0
            n += 1
        for i in range(m):
            tot += abs(x[i] - y[i]) / (abs(x[i]) + abs(y[i]) + 1e-9)
            n += 1
    return tot / max(n, 1)


def _informative(family, seed, difficulty) -> dict:
    base = make(family, seed, **difficulty)
    noise = _replicate(base, 0)
    other = make(family, seed + 1000, **difficulty)
    other = Env(family, seed + 1000, other._config, difficulty, world_seed=base.world_seed)
    s0, s1, s2 = _stream(_fork(base)), _stream(noise), _stream(other)
    dn, ds = _stream_distance(s0, s1), _stream_distance(s0, s2)
    return {"ok": ds > 1.5 * dn + 1e-6, "distance_between_instances": ds, "distance_between_noise_draws": dn}


def _axes_effective(family, seed) -> dict:
    fam = FAMILIES[family]
    base = digest(build_config(family, seed))
    out = {}
    for ax, v in fam.axes.items():
        nv = v * 1.5 if isinstance(v, float) else (int(v) + max(1, int(v) // 2) if v else 1)
        if ax == "observation_noise" and v == 0:
            nv = 0.5
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
    ck["achievable"] = {"ok": gap > max(3.0 * noise_sd, 1e-6) and v_or >= max(triv.values()), "oracle": v_or,
                        "default": v_def, "how": orc.get("how"), "objective": env._config["objective"]["type"]}
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
