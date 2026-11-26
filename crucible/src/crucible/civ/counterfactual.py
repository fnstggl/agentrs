"""Counterfactual experiments on civilisations: fork, intervene, rerun, compare across seeds.

Protocol (one experiment, several independent seeds):

1. build the civilisation of a seed and run it to the fork time `t0` (warm-up); take a snapshot;
2. from that snapshot run every arm to `t0 + horizon`: the interventions (remove a coupling, remove an entity or
   artifact, change a resource, let a scripted agent act or not) and two placebo arms that record `n` no-op events
   at the fork (`admin.links_placebo`) and change nothing else;
3. measure each arm with evaluator-side functions of the world (they may read hidden state: they are the
   experimenter's instruments, never shown to participants).

Why placebos: the kernel's random streams are keyed by event sequence numbers (`Ctx.rng`), so ANY intervention that
records events reshuffles every later random draw, and until workstream A's subscriber-RNG isolation lands so does
adding or removing an event subscriber (audit A_J, defect K2). Two placebo arms with different footprints
(1 and 5 no-op events) measure how much a quantity moves from reshuffling alone (`noise` = P1 - P5, per seed). An
effect is confirmed only when, across seeds, the intervention minus the placebo mean is consistently signed, its
paired 95% confidence interval excludes zero, and its mean exceeds twice the reshuffling noise (root mean square of
P1 - P5). The noise is reported with every effect as the measured confounding.

Experiments are named entries of `crucible.civ.experiments.EXPERIMENTS`, so a seed can run in its own process
(`python -m crucible.civ.counterfactual seed <name> <seed> <out.json>`): `run(name, seeds, workers=k)` runs k seeds
at a time (launch the caller through `crucible.scale.slots --cpu k`).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

PLACEBOS = {"placebo_1": 1, "placebo_5": 5}


@dataclass
class Experiment:
    name: str
    doc: str
    profile: str
    warmup_days: float
    horizon_days: float
    interventions: dict[str, Callable[[Any], None]]          # arm -> fn(world) applied at t0
    measures: dict[str, Callable[[Any, int, int, dict], float]]   # name -> fn(world, t0, t1, ctx) evaluator-side
    primary: str                                             # the measure that confirms the effect
    category: int | None = None                              # Part 10.2 category it demonstrates
    population: str = "development"
    overrides: dict | None = None
    setup: Callable[[Any, dict], None] | None = None         # at t0 in EVERY arm (e.g. admit an agent)
    during: Callable[[Any, str, dict], None] | None = None   # called hourly in every arm (e.g. an agent's policy)
    prelude: Callable[[Any, dict], None] | None = None       # before the warm-up (e.g. admit an agent)
    prelude_during: Callable[[Any, str, dict], None] | None = None   # hourly during the warm-up
    chain: bool = False                                      # mine chains in the reference arm
    reference_arm: str = "factual"
    seeds: tuple[int, ...] = (1, 2, 3)
    meta: dict = field(default_factory=dict)


def placebo(world, n: int) -> None:
    world.system_call("admin.links_placebo", {"n": int(n)})


def _arms(exp: Experiment) -> list[str]:
    return ["factual", *PLACEBOS, *exp.interventions]


def run_seed(exp: Experiment, seed: int, *, log=None) -> dict:
    """All arms of one seed: {arm: {measure: value}}, the fork time, footprints, wall times and (for chain
    experiments) the chains mined in the reference arm."""
    import torch
    torch.set_num_threads(1)
    from ..core.kernel import World
    from . import chains, generator
    t_start = time.time()
    w = generator.generate(exp.profile, seed, exp.population, overrides=exp.overrides)
    t0 = int(exp.warmup_days * 86400)
    if exp.prelude is not None:
        pctx: dict = {"arm": "prelude", "seed": seed, "t0": 0, "seq0": w.seq}
        exp.prelude(w, pctx)
        t = w.now
        while t < t0:
            t = min(t0, (t // 3600 + 1) * 3600)
            w.advance_time(t)
            if exp.prelude_during is not None:
                exp.prelude_during(w, "prelude", pctx)
    w.advance_time(t0)
    blob = w.snapshot_blob()
    base_seq = w.seq
    out = {"seed": seed, "t0": t0, "t1": t0 + int(exp.horizon_days * 86400), "arms": {}, "footprint": {},
           "wall_s": {}, "warmup_wall_s": round(time.time() - t_start, 1), "invariants": {}}
    for arm in _arms(exp):
        ta = time.time()
        wa = World.from_blob(blob, world_id=f"cf-{exp.name}-{seed}-{arm}")
        ctx: dict = {"arm": arm, "seed": seed, "t0": t0, "seq0": wa.seq}
        if exp.setup is not None:
            exp.setup(wa, ctx)
        s0 = wa.seq
        if arm in PLACEBOS:
            placebo(wa, PLACEBOS[arm])
        elif arm != "factual":
            exp.interventions[arm](wa)
        out["footprint"][arm] = wa.seq - s0
        t1 = out["t1"]
        if exp.during is not None:
            t = wa.now
            while t < t1:
                t = min(t1, (t // 3600 + 1) * 3600)
                wa.advance_time(t)
                exp.during(wa, arm, ctx)
        else:
            wa.advance_time(t1)
        out["arms"][arm] = {k: float(fn(wa, t0, t1, ctx)) for k, fn in exp.measures.items()}
        out["invariants"][arm] = wa.check_invariants()[:5]
        if exp.chain and arm == exp.reference_arm:
            ch = chains.mine(wa, from_seq=base_seq)
            out["chains"] = [_chain_record(c) for c in ch]
        out["wall_s"][arm] = round(time.time() - ta, 1)
        if log:
            log(f"{exp.name} seed {seed} arm {arm}: {out['arms'][arm]} ({out['wall_s'][arm]} s)")
        wa.close()
    w.close()
    return out


def _chain_record(c: dict) -> dict:
    return {k: c[k] for k in ("root_seq", "root_type", "root_t", "root_entity", "actor", "actor_kind", "scripted",
                              "n_stages", "agent_artifact", "social", "technical")} | {
        "stages": [{k: s[k] for k in ("subsystem", "first_seq", "first_t", "n_events", "types", "social",
                                      "technical")} for s in c["stages"]]}


# --------------------------------------------------------------------------------------------- statistics
def _t_crit(df: int) -> float:
    from scipy.stats import t
    return float(t.ppf(0.975, max(1, df)))


def analyze(exp: Experiment, per_seed: list[dict]) -> dict:
    """Per intervention arm and measure: paired effect against the placebo mean, CI, reshuffling noise, verdict."""
    res = {"experiment": exp.name, "category": exp.category, "profile": exp.profile, "population": exp.population,
           "seeds": [r["seed"] for r in per_seed], "primary": exp.primary, "effects": {}}
    for arm in exp.interventions:
        res["effects"][arm] = {}
        for m in exp.measures:
            p = np.array([np.mean([r["arms"][a][m] for a in PLACEBOS]) for r in per_seed], float)
            c = np.array([r["arms"][arm][m] for r in per_seed], float)
            f = np.array([r["arms"]["factual"][m] for r in per_seed], float)
            noise = np.array([r["arms"]["placebo_1"][m] - r["arms"]["placebo_5"][m] for r in per_seed], float)
            d = c - p
            n = d.size
            mean = float(d.mean()) if n else math.nan
            sd = float(d.std(ddof=1)) if n > 1 else math.nan
            half = _t_crit(n - 1) * sd / math.sqrt(n) if n > 1 else math.inf
            rms = float(np.sqrt(np.mean(noise ** 2))) if n else math.nan
            signs = np.sign(d)
            consistent = bool(n and (np.all(signs > 0) or np.all(signs < 0)))
            ci = [mean - half, mean + half]
            confirmed = bool(n >= 2 and consistent and (ci[0] > 0 or ci[1] < 0) and abs(mean) > 2 * rms)
            res["effects"][arm][m] = {
                "intervention": c.tolist(), "placebo_mean": p.tolist(), "factual": f.tolist(),
                "effect_mean": mean, "effect_sd": sd, "ci95": ci, "reshuffle_noise_rms": rms,
                "noise": noise.tolist(), "relative_effect": float(mean / np.mean(np.abs(p))) if np.mean(np.abs(p)) > 0
                else math.nan, "sign_consistent": consistent, "confirmed": confirmed}
        res["effects"][arm]["confirmed"] = bool(res["effects"][arm][exp.primary]["confirmed"])
    res["confirmed"] = any(res["effects"][a]["confirmed"] for a in exp.interventions)
    if exp.chain:
        res["chains"] = [r.get("chains", []) for r in per_seed]
    res["footprints"] = [r["footprint"] for r in per_seed]
    res["invariant_errors"] = sum(len(v) for r in per_seed for v in r["invariants"].values())
    res["wall_s"] = sum(sum(r["wall_s"].values()) + r["warmup_wall_s"] for r in per_seed)
    return res


# --------------------------------------------------------------------------------------------- running
def run(name: str, seeds=None, *, workers: int = 1, log=None, timeout_s: int = 6 * 3600) -> dict:
    """Run experiment `name` over `seeds` (default: its own) and analyze. With workers > 1 each seed runs in its own
    process (k at a time)."""
    from .experiments import EXPERIMENTS
    exp = EXPERIMENTS[name]
    seeds = list(seeds or exp.seeds)
    per = []
    if workers <= 1:
        for s in seeds:
            per.append(run_seed(exp, s, log=log))
    else:
        tmp = tempfile.mkdtemp(prefix="civcf-")
        pending = list(seeds)
        running: list[tuple[subprocess.Popen, str, int]] = []
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        while pending or running:
            while pending and len(running) < workers:
                s = pending.pop(0)
                path = os.path.join(tmp, f"{name}-{s}.json")
                pr = subprocess.Popen([sys.executable, "-m", "crucible.civ.counterfactual", "seed", name, str(s), path],
                                      env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                running.append((pr, path, s))
            pr, path, s = running.pop(0)
            try:
                _, err = pr.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                pr.kill()
                raise RuntimeError(f"{name} seed {s} timed out")
            if pr.returncode != 0:
                raise RuntimeError(f"{name} seed {s} failed: {err.decode(errors='replace')[-2000:]}")
            with open(path) as fh:
                per.append(json.load(fh))
            if log:
                log(f"{name} seed {s} done")
        per.sort(key=lambda r: seeds.index(r["seed"]))
    out = analyze(exp, per)
    out["per_seed"] = per
    return out


def _main(argv):
    if len(argv) >= 4 and argv[0] == "seed":
        from .experiments import EXPERIMENTS
        exp = EXPERIMENTS[argv[1]]
        r = run_seed(exp, int(argv[2]), log=lambda m: print(m, file=sys.stderr, flush=True))
        with open(argv[3], "w") as fh:
            json.dump(r, fh)
        return 0
    if len(argv) >= 2 and argv[0] == "run":
        seeds = [int(x) for x in argv[2].split(",")] if len(argv) > 2 else None
        workers = int(argv[3]) if len(argv) > 3 else 1
        r = run(argv[1], seeds, workers=workers, log=lambda m: print(m, file=sys.stderr, flush=True))
        r.pop("per_seed", None)
        print(json.dumps(r, indent=1, default=str))
        return 0
    print("usage: python -m crucible.civ.counterfactual seed <name> <seed> <out.json> | run <name> [seeds] [workers]")
    return 2


if __name__ == "__main__":
    import crucible
    crucible.load_all_modules()
    sys.exit(_main(sys.argv[1:]))
