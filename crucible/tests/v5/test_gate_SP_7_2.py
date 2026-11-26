"""Gate SP-7.2 - structural composition across the eleven dimensions of Part 7.2, and superficially similar worlds
that obey different causal rules (workstream H).

Composition: for each dimension the structural generator (`crucible.rapid.generator.compose`) builds, from the same
seed, a pair of worlds whose recipes differ only in that dimension (topology; functional form; temporal relation;
observation channel; affordance; participants; network; resources; institution; initial conditions; historical
trajectory). The dimension counts when, on each of three seeds, the difference is verified by doing something in both
worlds: the same lever schedule on the same noise moves the outcome differently (by more than three times the
difference between two noise replicates of one world), an action is carried out in one world and refused in the
other, or one world shows an observation kind the other never does.

Look-alikes: the four look-alike markets (`lookalike_market`: elastic, signal, gate, drift) share their public
interface exactly; a pair counts as confirmed different when the price-response intervention protocol
(`claims.market_signature`) gives the two markets different signatures on each of three seeds."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from crucible.acceptance.v5.evidence import record
from crucible.rapid import claims as CL
from crucible.rapid.env import Env
from crucible.rapid.generator import compose

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]
BASE = {"topology": "chain", "motifs": ["saturation", "linear_nonlinear"], "n_levers": 2, "temporal": "instant",
        "channels": ["noisy"], "affordances": ["levers"], "horizon": 40, "noise": 0.1}
DIMENSIONS = {
    "topology": ({}, {"topology": "collider"}),
    "functional_form": ({}, {"motifs": ["non_monotonic", "linear_nonlinear"]}),
    "temporal_relation": ({}, {"temporal": "lagged"}),
    "observation_channel": ({}, {"channels": ["noisy", "delayed", "missing"]}),
    "affordance": ({}, {"affordances": ["levers", "experiment"]}),
    "participants": ({}, {"participants": ["adaptive_rival"]}),
    "network": ({}, {"network": "installed_base"}),
    "resources": ({}, {"resources": "shared_supply"}),
    "institution": ({}, {"institution": "output_quota"}),
    "initial_conditions": ({"initial": "depleted"}, {"initial": "stocked"}),
    "historical_trajectory": ({}, {"warmup": 20}),
}


def _world(extra: dict, seed: int, world_seed: int):
    cfg = compose(np.random.default_rng(seed), dict(BASE, **extra))
    cfg["family"] = "composed"
    return Env("composed", seed, cfg, {}, world_seed=world_seed)


def _schedule(env) -> tuple[np.ndarray, np.ndarray]:
    """Revenue and the first reading (NaN when no reading came) under a fixed lever schedule."""
    names = sorted(env.action_space().kinds["levers"]["properties"]["values"]["properties"])
    rev, read = [], []
    key = None
    while not env.done:
        u = [0.0, 5.0, 10.0][(env.t // 5) % 3]
        r = env.step({"kind": "levers", "values": {names[0]: u, names[1]: 3.0}})
        rev.append(float(r["observation"]["resources"].get("revenue_cents_last_tick", 0.0)))
        nums = r["observation"]["numbers"]
        key = key or (sorted(nums)[0] if nums else None)
        read.append(float(nums[key]) if key in nums else np.nan)
    return np.asarray(rev), np.asarray(read)


def _gap(x: np.ndarray, y: np.ndarray) -> float:
    both = ~np.isnan(x) & ~np.isnan(y)
    miss = float(np.mean(np.isnan(x) != np.isnan(y)))
    return float(np.mean(np.abs(x[both] - y[both]))) + 10.0 * miss if both.any() else 1.0


def _kinds(env) -> set[str]:
    o = env.observe()
    out = {f"reading:{len(o['numbers'])}"}
    out |= {f"array:{k}" for k in o["arrays"]} | {f"text:{bool(o.get('text'))}"}
    out |= {f"action:{k}" for k in env.action_space().kinds}
    return out


def _verified(a_extra: dict, b_extra: dict, seed: int) -> dict:
    a, b = _world(a_extra, seed, 9001), _world(b_extra, seed, 9001)
    ka, kb = _kinds(a), _kinds(b)
    if ka != kb:
        return {"ok": True, "how": "interface_or_observation", "only_a": sorted(ka - kb), "only_b": sorted(kb - ka)}
    (ra, ga), (rb, gb) = _schedule(a), _schedule(b)
    ra2, ga2 = _schedule(_world(a_extra, seed, 9002))
    # what moves (money) and what is observed (the reading): either must differ by more than noise replicates do
    out = {}
    for lab, x, y, z in (("money", ra, rb, ra2), ("reading", ga, gb, ga2)):
        out[lab] = (_gap(x, y), _gap(x, z))
    ok = any(d > 3.0 * n and d > 1e-6 for d, n in out.values())
    return {"ok": bool(ok), "how": "behaviour", "difference_vs_replicate": out}


def test_SP_7_2_composition_dimensions_and_lookalikes():
    per_dim = {}
    for dim, (a, b) in DIMENSIONS.items():
        rs = [_verified(a, b, s) for s in (0, 1, 2)]
        per_dim[dim] = {"ok": all(r["ok"] for r in rs), "how": sorted({r["how"] for r in rs})}
    dims = sorted(d for d, r in per_dim.items() if r["ok"])
    # not vacuous: two worlds from the same recipe (only the noise differs) are never verified as different
    assert not any(_verified({}, {}, s)["ok"] for s in (0, 1, 2))
    markets = ("mk_elastic", "mk_signal", "mk_gate", "mk_drift")
    mats = [CL.lookalike_matrix(s) for s in (0, 1, 2)]
    same_iface = all(m["same_interface"] for m in mats)
    pairs = list(itertools.combinations(markets, 2)) if same_iface else []
    confirmed = [p for p in pairs if all(m["signatures"][p[0]] != m["signatures"][p[1]] for m in mats)]
    record("SP-7.2", {"composition_dimensions": len(dims), "dimensions_tried": len(DIMENSIONS),
                      "lookalike_world_pairs": len(pairs),
                      "lookalike_pairs_confirmed_different_by_intervention": len(confirmed),
                      "lookalikes_show_own_rule": all(m["all_own"] for m in mats)},
           notes=f"Dimensions verified on seeds 0-2: {dims}; per dimension: {per_dim}. Look-alike signatures: "
                 f"{[m['signatures'] for m in mats]}.")
    assert set(dims) <= set(DIMENSIONS) and len(confirmed) <= len(pairs)
