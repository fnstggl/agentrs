"""Category-level behavioural requirements of Parts 8.2-8.10 (evaluator side), measured on worlds of admitted families.

    unobserved_conditions(family, seed)      8.2  success is judged at conditions no participant could set or observe:
                                                  every held-out condition of the objective lies outside what the
                                                  action interface can set (levers, experiment designs, the probe box)
    action_forms(families)                   8.4  the families together offer discrete, continuous and programmatic
                                                  actions
    discovery_separation(family, seed)       8.5  without any reward, what a participant observes separates useful
                                                  discovery from random novelty: the family's public reference and a
                                                  random explorer of the same action kinds run on the same world,
                                                  and the income each one *observes* (its own resources) differs by
                                                  more than the noise of replicate runs
    opponent_populations(train, eval)        8.8  training families face a population of opponent policy families,
                                                  and every evaluation family faces at least one opponent policy family
                                                  no training family has (held out)
    interdependence(family, seed)            8.9  independent participants cannot succeed: one adaptive learner per seat
                                                  acting on its own observed income, with no messages and no joint
                                                  plans, gains at most half of what the family's coordinating
                                                  reference gains
    constants_fail_after_change(family, seed) 8.10 the constant setting that did best before the change does clearly
                                                  worse afterwards than the setting that does best afterwards (common
                                                  random numbers, a fixed set of constant settings); for families
                                                  scored by held-out predictions, on the scored channel itself: the
                                                  predictions that were exactly right as the world stood at the end
                                                  of the first fifth are clearly wrong at the end
                                                  (`beliefs_fail_after_change`)

Each function returns a JSON-able record with "ok". Nothing here is imported by world code.
"""
from __future__ import annotations

import math

import numpy as np

from . import dominance as D
from .families import make
from .identify import PublicView
from .oracles import fork

PROGRAMMATIC = {"program"}
DISCRETE = {"discrete", "graph_edit", "coordinate"}
CONTINUOUS = {"levers", "allocate"}


# ====================================================================== 8.2
def _settable(env) -> dict[str, tuple[float, float]]:
    sp = env.action_space()
    out: dict[str, tuple[float, float]] = {}
    if "levers" in sp.kinds:
        for n, p in sp.kinds["levers"]["properties"]["values"]["properties"].items():
            out[n] = (float(p["minimum"]), float(p["maximum"]))
    if "experiment" in sp.kinds:
        props = sp.kinds["experiment"]["properties"]["design"]["properties"].get("set", {}).get("properties", {})
        for n, p in props.items():
            lo, hi = float(p.get("minimum", -math.inf)), float(p.get("maximum", math.inf))
            a, b = out.get(n, (lo, hi))
            out[n] = (min(a, lo), max(b, hi))
    for n, (lo, hi) in (env._config.get("box") or {}).items():
        a, b = out.get(n, (float(lo), float(hi)))
        out[n] = (min(a, float(lo)), max(b, float(hi)))
    return out


def _heldout(env) -> list[dict]:
    c = env._config
    obj = c.get("objective") or {}
    if obj.get("heldout"):
        return list(obj["heldout"])
    if c.get("heldout"):
        return list(c["heldout"])
    return list((c.get("hidden") or {}).get("holdout") or [])


def unobserved_conditions(family: str, seed: int = 0) -> dict:
    env = make(family, seed)
    obj = (env._config.get("objective") or {}).get("type")
    rng = _settable(env)
    held = _heldout(env)
    outside = []
    for cond in held:
        out = any(isinstance(v, (int, float)) and k in rng and not (rng[k][0] - 1e-9 <= float(v) <= rng[k][1] + 1e-9)
                  for k, v in cond.items())
        outside.append(bool(out))
    ok = obj in ("predict_heldout", "heldout_tests") and bool(held) and all(outside)
    return {"ok": bool(ok), "objective": obj, "heldout_conditions": len(held),
            "outside_settable_range": int(sum(outside))}


# ====================================================================== 8.4
def action_forms(families: list[str]) -> dict:
    kinds: set[str] = set()
    for n in families:
        env = make(n, 0)
        for a in env.agents():
            kinds |= set(env.action_space(a).kinds)
    forms = {"discrete": bool(kinds & DISCRETE), "continuous": bool(kinds & CONTINUOUS),
             "programmatic": bool(kinds & PROGRAMMATIC)}
    return {"ok": all(forms.values()), **forms, "kinds": sorted(kinds)}


# ====================================================================== 8.5
def _observed_income(env) -> float:
    """Income the participant observed tick by tick (revenue minus cost in its own resources) - the very numbers
    the environment's ledger books for it, never the evaluator's outcome."""
    a = env.agents()[0]
    return float(env._ledger["revenue"][a] - env._ledger["cost"][a])


def discovery_separation(family: str, seed: int = 0, replicates: int = 5) -> dict:
    """The family's public reference (useful discovery) and a random explorer of the same action kinds (novelty:
    new cells dug, new queries, new options, with no model of where value lies) on replicate worlds; no reward is
    ever returned (checked on every step)."""
    from .admission import _replicate
    from .policies import random_policy
    from .references import play
    base = make(family, seed)
    ref, rnd = [], []
    for r in range(replicates):
        e = _replicate(base, r)
        f = fork(e)
        play(f)
        ref.append(_observed_income(f))
        g = fork(e)
        rng = np.random.default_rng(100 + r)
        obs = g.observe()
        while not g.done:
            out = g.step(random_policy(obs, g.action_space(), rng))
            obs = out["observation"]
            assert "reward" not in out and "reward" not in obs
        rnd.append(_observed_income(g))
    gap = float(np.mean(ref) - np.mean(rnd))
    se = float(np.sqrt(np.var(ref, ddof=1) / len(ref) + np.var(rnd, ddof=1) / len(rnd))) + 1e-9
    return {"ok": bool(gap > 0 and gap / se > 3.0), "observed_income_discovery": float(np.mean(ref)),
            "observed_income_random_novelty": float(np.mean(rnd)), "gap": gap, "z": gap / se}


# ====================================================================== 8.8
def _opponents(family: str) -> set[str]:
    from .fingerprints import family_fingerprint
    return {p for p in family_fingerprint(family)["participants"] if not p.startswith("seats:")}


def opponent_populations(train: list[str], evals: list[str]) -> dict:
    pool = set().union(*[_opponents(n) for n in train]) if train else set()
    held = {n: sorted(_opponents(n) - pool) for n in evals}
    ok = len(pool) >= 3 and bool(evals) and all(held.values())
    return {"ok": bool(ok), "training_opponent_policy_families": sorted(pool),
            "evaluation_held_out_opponents": held}


# ====================================================================== 8.9
def interdependence(family: str, seed: int = 0) -> dict:
    from .references import play
    from .refs_v2 import adaptive
    v_wait = float(D.play(make(family, seed), lambda t, o: {})[0])
    v_ref = float(play(make(family, seed))["value"])
    e = make(family, seed)
    adaptive(PublicView(e), e)                     # one learner per seat, no messages, no joint plans
    v_ind = float(e.outcome()["value"])
    g_ref, g_ind = v_ref - v_wait, v_ind - v_wait
    return {"ok": bool(g_ref > 0 and g_ind <= 0.5 * g_ref), "reference_gain": g_ref,
            "independent_learners_gain": g_ind, "seats": len(e.agents())}


# ====================================================================== 8.10
def beliefs_fail_after_change(family: str, seed: int = 0) -> dict:
    """8.10 for a family scored by predictions at held-out conditions (primitive-graph engine): the scored truth is
    the noise-free response under the conditions at the end of the episode. The same truth computed for the world
    as it stood at the end of the first fifth (`true_response(..., t_frozen, exogenous_ticks)`: timed regimes
    frozen there, the world's own drifting stocks held where they were then, everything the levers drive still
    settling) is what a learner that had learned the early world perfectly would file. Filed as
    predictions, it must fall clearly short: its shortfall below perfect predictions is at least a quarter of the
    gain of perfect predictions over filing nothing (the score is 1 - MSE / variance of the truth; filing nothing
    scores 0). Measured on the channel the family is scored on, so parameters that never enter the score (lever
    costs, revenue) cannot decide it."""
    from .engines.primitive_graph import score_predictions, true_response
    env = make(family, seed)
    obj = env._config["objective"]
    eng = env._engine
    H = env.horizon
    k = max(3, H // 5)
    ticks = int(obj.get("settle_ticks", 1))
    early = true_response(eng, obj["heldout"], obj["gauge"], ticks=ticks, t_frozen=k - 1, exogenous_ticks=k)
    journal = [{"content": {"predictions": [{"set": c, "value": float(v)} for c, v in zip(obj["heldout"], early)]}}]
    score = float(score_predictions(eng, journal)["prediction_score"])
    shortfall = 1.0 - score
    return {"ok": bool(shortfall >= 0.25), "relative_shortfall_of_early_truth": shortfall,
            "early_truth_score": score, "frozen_at_tick": k - 1, "channel": "scored predictions"}


def _scored_by_predictions(env) -> bool:
    from .engines.primitive_graph import PrimitiveGraph
    return (env._config.get("objective") or {}).get("type") == "predict_heldout" and \
        isinstance(env._engine, PrimitiveGraph)


def constants_fail_after_change(family: str, seed: int = 0, n_random: int = 16) -> dict:
    """The constant setting (from a fixed set: the canonical grid of the action space and 16 random settings) that
    earns most in the first fifth of the episode, held on, earns clearly less than the best constant in some later
    fifth: its shortfall there is at least a quarter of that fifth's best gain over waiting (common random
    numbers; observed net income per tick). Families scored by held-out predictions are measured on their scored
    channel instead (`beliefs_fail_after_change`; verifier finding on the wave-3b tree: net income of constants
    measured a channel those families are not scored on, decided by a lever cost that never enters the score)."""
    env = make(family, seed)
    if _scored_by_predictions(env):
        return beliefs_fail_after_change(family, seed)
    joint = D.Joint(env)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 4243]))
    settings = joint.canonical() + [joint.random(rng) for _ in range(n_random)]
    H = env.horizon
    k = max(3, H // 5)

    def per_tick(pol):
        e = fork(env)
        ags = e.agents()
        obs = {a: e.observe(a) for a in ags}
        out = []
        while not e.done:
            acts = pol(e.t, obs)
            r = e.step(acts if len(ags) > 1 else acts.get(ags[0]))
            obs = r.get("observations") or {ags[0]: r["observation"]}
            out.append(D._net(obs[ags[0]]))
        return np.asarray(out)
    wait = per_tick(lambda t, o: {})
    rows = [per_tick(D.constant(joint, s)) for s in settings]
    first = int(np.argmax([r[2:k].mean() for r in rows]))
    best_loss, where = 0.0, None
    for j in range(1, H // k):
        a, b = j * k, min(H, (j + 1) * k)
        vals = [r[a:b].mean() for r in rows]
        bj = int(np.argmax(vals))
        gain = vals[bj] - float(wait[a:b].mean())
        if gain > 1e-9:
            loss = (vals[bj] - vals[first]) / gain
            if loss > best_loss:
                best_loss, where = float(loss), j
    return {"ok": bool(best_loss >= 0.25), "relative_shortfall_of_early_best": best_loss, "in_fifth": where,
            "early_best_setting": first, "settings_tried": len(settings)}
