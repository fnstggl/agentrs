"""The world-proposal interface (MASTER_SPEC Part 14.4; SP-14.4).

A research learner proposes a new experimental world as a *structure* of the generator's grammar
(`compose.normalize`): JSON data only, never code. It sends it on its participant channel (``propose``); the
evaluator process validates it against its own registry and returns only a public verdict:

    {"accepted": bool, "structure_id", "checks": {check: bool}, "reasons": [public strings], "measured": {...}}

Six checks, all measured evaluator side on copies of the proposed world:

    safety        a JSON object of at most `MAX_BYTES` with only grammar fields (no partition, split, family, seed,
                  evaluator or file fields; every string an enumerated grammar token); the grammar's own bounds on
                  sizes and horizon
    feasibility   the world builds and calibrates on `SEEDS`, and the evaluator-side upper reference improves on
                  leaving the defaults alone by at least `MIN_HEADROOM` of the defaults' net on every seed
    diversity     measured structural fingerprint distance > 0 to every training family, every evaluation family
                  and every proposal accepted before (a renamed copy of an existing world is not new)
    resources     measured CPU seconds of one episode and the size of the world configuration within budget
    difficulty    the best canonical constant setting (all levers at minimum / middle / maximum, the defaults)
                  reaches less than `TRIVIAL_MAX` of the upper reference's improvement (no trivial solution) and the
                  scripted adaptive reference reaches more than `HOPELESS_MIN` (not hopeless)
    usefulness    the structure adds a structural element (a motif, motif-edge, observation channel, temporal or
                  participant descriptor) that no training family has, or targets a weakness category given by the
                  caller's weakness profile

An accepted proposal joins the evaluator's *training* distribution only (`Registry.extend`: generated entries are
training families; evaluation and sealed partitions cannot be extended). Nothing in the verdict carries
evaluator internals beyond pass/fail and coarse public reasons.
"""
from __future__ import annotations

import json
import time
from typing import Any, Mapping, Sequence

import numpy as np

from . import compose as C
from .distribution import Registry, entry_fingerprint, generated_entry, thaw

CHECKS = ("safety", "feasibility", "diversity", "resources", "difficulty", "usefulness")
MAX_BYTES = 4096
FORBIDDEN_FIELDS = ("split", "partition", "family", "seed", "seeds", "sealed", "evaluator", "oracle", "reference",
                    "path", "file", "code", "url", "salt", "name")
SEEDS = (0, 1)
MIN_HEADROOM = 0.02
MAX_EPISODE_CPU_S = 1.0
MAX_CONFIG_BYTES = 400_000
TRIVIAL_MAX = 0.8
HOPELESS_MIN = 0.02


def _strings(x, out: list) -> None:
    if isinstance(x, str):
        out.append(x)
    elif isinstance(x, Mapping):
        for k, v in x.items():
            out.append(str(k))
            _strings(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _strings(v, out)


def _fp_tokens(fp: Mapping) -> set:
    out = set()
    for k in ("motifs", "motif_edges", "observation", "temporal", "participants", "topology"):
        v = fp.get(k)
        if isinstance(v, (list, tuple, set)):
            out |= {f"{k}:{x}" for x in v}
        elif v is not None:
            out.add(f"{k}:{v}")
    return out


def validate_proposal(reg: Registry, proposal: Any, *, proposer: str = "learner", accepted: Sequence[str] = (),
                      weakness: Mapping[str, float] | None = None) -> dict:
    checks = {c: False for c in CHECKS}
    reasons: list[str] = []
    measured: dict = {}
    out = {"accepted": False, "structure_id": None, "checks": checks, "reasons": reasons, "measured": measured,
           "proposer": proposer}
    # ---- safety
    try:
        raw = json.dumps(proposal)
    except (TypeError, ValueError):
        reasons.append("safety: a proposal is JSON data")
        return out
    if len(raw.encode()) > MAX_BYTES:
        reasons.append(f"safety: larger than {MAX_BYTES} bytes")
        return out
    if not isinstance(proposal, Mapping):
        reasons.append("safety: a proposal is a JSON object")
        return out
    bad = sorted({k for k in proposal if k.lower() in FORBIDDEN_FIELDS})
    if bad:
        reasons.append(f"safety: fields {bad} are not part of a world structure (partitions, seeds and evaluator "
                       "settings cannot be proposed)")
        return out
    try:
        s = C.normalize(dict(proposal))
    except C.CompositionError as e:
        reasons.append(f"safety: not in the generator grammar ({str(e)[:160]})")
        return out
    toks: list[str] = []
    _strings(s, toks)
    allowed = set(C.TOPOLOGIES) | set(C.MOTIFS) | set(C.COMBINERS) | set(C.TEMPORAL) | set(C.CHANNELS) | \
        set(C.DRIVERS) | set(C.POPULATIONS) | {"undercut", "match", C.COMPOSE_VERSION, "topology", "motifs",
                                               "combiner", "temporal", "channels", "n_levers", "drivers",
                                               "population", "horizon", "version", "kind", "n", "capability",
                                               "mode"}
    if any(t not in allowed for t in toks):
        reasons.append("safety: free text is not allowed in a structure")
        return out
    checks["safety"] = True
    sid = C.structure_id(s)
    out["structure_id"] = sid
    entry = generated_entry(s, proposer=proposer)
    # ---- feasibility and resources and difficulty (measured on copies)
    from .episodes import references, run_episode
    from .learners import make as make_learner
    from .sampler import build_env, rapid_world
    feas, cpu, cfg_bytes, trivial, hopeless = True, 0.0, 0, [], []
    try:
        for sd in SEEDS:
            cfg = rapid_world(entry, sd, {}, None)._config
            cfg_bytes = max(cfg_bytes, len(json.dumps(cfg, default=str)))
            if not cfg.get("calibrated"):
                feas = False
                reasons.append("feasibility: no interior calibration exists for this structure")
                break
            refs = references(entry, sd)
            head = (refs["upper"] - refs["wait"]) / max(abs(refs["wait"]), 1.0)
            measured.setdefault("headroom", []).append(round(float(head), 4))
            if head < MIN_HEADROOM:
                feas = False
                reasons.append("feasibility: nothing can be improved over leaving the world alone")
                break
            t0 = time.process_time()
            env = build_env(entry, sd, {}, {"mode": "assigned"})
            rec = run_episode(env, make_learner("adapter", 0), learn=False, rng=np.random.default_rng(sd))
            cpu = max(cpu, time.process_time() - t0)
            span = refs["upper"] - refs["wait"]
            hopeless.append((rec["value"] - refs["wait"]) / span)
            best = -np.inf
            for frac in (0.0, 0.5, 1.0, None):
                e2 = build_env(entry, sd, {}, {"mode": "assigned"})
                L = _Const(frac)
                r2 = run_episode(e2, L, learn=False, rng=np.random.default_rng(sd))
                best = max(best, (r2["value"] - refs["wait"]) / span)
            trivial.append(best)
    except Exception as e:  # noqa: BLE001 - a world that cannot be built is not feasible
        feas = False
        reasons.append(f"feasibility: the world could not be built ({type(e).__name__})")
    checks["feasibility"] = feas
    measured["episode_cpu_s"] = round(cpu, 4)
    measured["config_bytes"] = cfg_bytes
    checks["resources"] = feas and cpu <= MAX_EPISODE_CPU_S and cfg_bytes <= MAX_CONFIG_BYTES
    if feas and not checks["resources"]:
        reasons.append("resources: an episode costs too much compute or memory")
    if feas:
        measured["trivial_share"] = round(float(max(trivial)), 4)
        measured["reference_share"] = round(float(np.mean(hopeless)), 4)
        checks["difficulty"] = max(trivial) < TRIVIAL_MAX and float(np.mean(hopeless)) > HOPELESS_MIN
        if not checks["difficulty"]:
            reasons.append("difficulty: solved by a canonical constant setting" if max(trivial) >= TRIVIAL_MAX
                           else "difficulty: the scripted adaptive reference cannot improve on waiting")
    # ---- diversity and usefulness (measured fingerprints)
    try:
        fp = entry_fingerprint(entry)
        from ...rapid import fingerprints as FP
        nearest, dmin = None, float("inf")
        train_tokens: set = set()
        for w in reg.worlds(tier=1):
            if w["source"] not in ("crucible.rapid", "generated") or w["id"] == entry["id"]:
                continue
            fw = entry_fingerprint(w)
            d = FP.distance(fp, fw)
            if d < dmin:
                nearest, dmin = w["id"], d
            if w["family_partition"] == "train":
                train_tokens |= _fp_tokens(fw)
        dup = sid in set(accepted)
        checks["diversity"] = dmin > 0.0 and not dup
        measured["nearest_distance"] = round(float(dmin), 4)
        if not checks["diversity"]:
            reasons.append("diversity: structurally identical to an existing world" if not dup else
                           "diversity: already proposed and accepted")
        new = sorted(_fp_tokens(fp) - train_tokens)
        from .curriculum import category_of
        cat = category_of(s)
        targeted = bool(weakness) and float((weakness or {}).get(cat, 0.0)) > 0.0
        checks["usefulness"] = bool(new) or targeted
        measured["new_structural_elements"] = len(new)
        measured["category"] = cat
        if not checks["usefulness"]:
            reasons.append("usefulness: adds no structural element and targets no measured weakness")
    except Exception as e:  # noqa: BLE001
        reasons.append(f"diversity: fingerprint could not be measured ({type(e).__name__})")
    out["accepted"] = all(checks.values())
    if out["accepted"]:
        out["_entry"] = entry
    return out


class _Const:
    """A canonical constant setting (fraction of every lever's range, or the defaults): the trivial strategies."""
    frozen = True

    def __init__(self, frac):
        self.frac = frac

    def begin(self, ctx, rng):
        self.u = None if self.frac is None else np.full(ctx["n"], float(self.frac))

    def act(self, k):
        return self.u

    def observe(self, k, s):
        pass

    def end(self, learn):
        pass
