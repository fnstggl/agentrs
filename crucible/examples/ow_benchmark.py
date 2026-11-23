"""The OpenWorld learner benchmark: one command per experiment.

    python examples/ow_benchmark.py baseline        --learners hold,random,probe_exploit,hill_climb
    python examples/ow_benchmark.py lifetime        --learners ppo_gru,model_based --seeds 0,1,2 --budget 400000
    python examples/ow_benchmark.py compare         --learners ppo_gru,ppo_gru_flat,probe_exploit --reference probe_exploit
    python examples/ow_benchmark.py timescales      --learners ppo_gru,model_based --budget 400000
    python examples/ow_benchmark.py internalisation --learners ppo_gru --budget 400000 --second-families abm_utility
    python examples/ow_benchmark.py transfer        --learners ppo_gru --budget 400000 --adapt-budget 100000
    python examples/ow_benchmark.py sealed          --learners ppo_gru,probe_exploit --sealed-set final-1 \\
                                                    --purpose "final table of the 4.0 report" --ack "<acknowledgement>"
    python examples/ow_benchmark.py report          --tag mytag          # re-render every result of a tag
    python examples/ow_benchmark.py list                                 # learners, families and sets available

Learners are named on the command line and resolved through the registry (`crucible.research.learners`); options
for one go in `--opts '{"ppo_gru": {"hidden": 32}}'`. Experiments are the protocols of `crucible.research.protocols`:

    baseline         frozen evaluation of the learners as built, on the public evaluation sets (paired, corrected)
    lifetime         a learning curve per (learner, seed): training on the training families, frozen evaluations
                     on every evaluation set, all costs, checkpoints
    compare          the same runs as `lifetime` (they are shared), compared under equal budgets, with the failure
                     diagnosis (`--kernel-diagnose N` also replays up to N failed tasks per learner through tool
                     calls in a kernel world and classifies the trajectory)
    timescales       fast adaptation, parametric learning and meta-learning per (learner, seed), measured apart
    internalisation  the eight-step parametric internalisation protocol per (learner, seed), optional retention
    transfer         zero-shot and adaptation transfer to the four categories per learner
    sealed           the final test: the learners' last checkpoints from `lifetime` (or as built, if they have
                     none) on a sealed set - one access for all of them, logged, budgeted

Evaluation sets are public validation tasks: `within` (new seeds of the training families) and one set per
relation of a validation family to the training set (`compositional`, `structural`, `cross_impl`).

Everything is resumable. Each (experiment, learner, seed) is a unit whose result is written when it finishes and
reused afterwards; learning curves additionally checkpoint at every point and continue from the last one. Units
run in `--workers` processes. `--fresh` discards the units of this experiment and tag first. Results go to
`docs/experiments/ow_benchmark_<experiment>_<tag>.json`, the report (Markdown and one PNG per quantity) to
`docs/figures/ow_benchmark_<experiment>_<tag>/` (`--no-report` leaves it to a later `report`), working files
(units, checkpoints) to `--work`.
"""
import argparse
import json
import os
import shutil
import sys
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

EXPERIMENTS = ("baseline", "lifetime", "compare", "timescales", "internalisation", "transfer", "sealed")


# ====================================================================== pieces shared by parent and workers
def _spec():
    from crucible.research.learner import EnvSpec
    return EnvSpec()


def _factory(name: str, cfg: dict):
    """A learner factory resolved through the registry, and nothing else."""
    from crucible.research import learners as registry
    from crucible.research import protocols as P
    if name not in registry.available():
        raise SystemExit(f"unknown learner {name!r}; the registry has: {', '.join(registry.available())}")
    return P.from_registry(name, _spec(), cfg["device"], **cfg["opts"].get(name, {}))


def _train_source(cfg: dict, families=None, start: int = 0):
    """Training tasks: a pool of `train_pool` tasks per family cycled without end (cached on disk by the learners'
    own helper when it is available), or an endless index-addressable stream when the pool size is 0."""
    from crucible.research import protocols as P
    fams = list(families or cfg["train_families"])
    if cfg["train_pool"] > 0:
        try:
            from crucible.research.learners.training import train_pool
            return train_pool(fams, cfg["train_pool"], start=start, workers=1)
        except ImportError:
            from crucible.research.tasks import make_task
            return [make_task(f, start + s, validate=False) for s in range(cfg["train_pool"]) for f in fams]
    return P.task_stream(fams, start)


def _eval_sets(cfg: dict, workers: int = 1) -> dict:
    """The public evaluation sets (validated tasks from the validation seed range, cached on disk)."""
    from crucible.research import tasks as T
    sets = {}
    if cfg["train_families"]:
        sets["within"] = T.eval_tasks(cfg["train_families"], cfg["n_eval"], workers=workers, skip_failures=True)
    for kind, fams in cfg["eval_families"].items():
        if fams:
            sets[kind] = T.eval_tasks(fams, cfg["n_eval"], workers=workers, skip_failures=True)
    return {k: v for k, v in sets.items() if v}


def _unit_path(cfg: dict, experiment: str, learner: str, seed) -> str:
    return os.path.join(cfg["work"], cfg["tag"], experiment, "units", f"{learner}__seed{seed}.json")


def _run_unit(job: dict) -> dict:
    """One (experiment, learner, seed). Runs in a worker process; the result file is the hand-over."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    from crucible.research import protocols as P
    cfg, exp, name, seed = job["cfg"], job["experiment"], job["learner"], job["seed"]
    path = _unit_path(cfg, exp, name, seed)
    if os.path.exists(path):
        return {"path": path, "learner": name, "seed": seed, "status": "reused", "wall_s": 0.0}
    t0 = time.time()
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:                                    # noqa: BLE001 - torch is optional
        pass
    factory = _factory(name, cfg)
    sets = _eval_sets(cfg)
    spec = _spec()
    wd = os.path.join(cfg["work"], cfg["tag"], exp, "checkpoints", f"{name}__seed{seed}")
    common = dict(spec=spec, n_envs=cfg["n_envs"], segments=cfg["segments"])
    if exp == "baseline":
        tasks = [t for ts in sets.values() for t in ts]
        res = P.evaluate(factory(seed), tasks, label=f"{name}/seed{seed}", **common)
    elif exp == "lifetime":
        res = P.lifetime(factory(seed), _train_source(cfg), cfg["budget"], cfg["eval_every"], sets,
                         checkpoint_dir=wd, require_disjoint=True, thresholds=tuple(cfg["thresholds"]),
                         sustain=cfg["sustain"], label=f"{name}/seed{seed}", **common)
    elif exp == "timescales":
        tasks = [t for ts in sets.values() for t in ts]
        res = P.timescales(factory, _train_source(cfg), tasks, cfg["budget"], seed=seed, k_reset=cfg["k_reset"],
                           workdir=wd, label=f"{name}/seed{seed}", **common)
    elif exp == "internalisation":
        tasks = [t for ts in sets.values() for t in ts]
        ret = None
        if cfg["second_families"]:
            ret = {"train_tasks": _train_source(cfg, cfg["second_families"]),
                   "budget_env_steps": cfg["second_budget"] or cfg["budget"]}
        res = P.internalisation(factory, _train_source(cfg), tasks, cfg["budget"], seed=seed, retention=ret,
                                workdir=wd, label=f"{name}/seed{seed}", **common)
    elif exp == "transfer":
        targets = {}
        for kind, ts in sets.items():
            fams = cfg["train_families"] if kind == "within" else cfg["eval_families"][kind]
            targets[kind] = {"category": kind, "eval": ts,
                             "train": _train_source(cfg, fams, start=cfg["adapt_start"]) if cfg["adapt_budget"] else None}
        res = P.transfer(factory, _train_source(cfg), targets, source_budget=cfg["budget"],
                         adapt_budget=cfg["adapt_budget"], eval_every=cfg["adapt_eval_every"], seeds=job["seeds"],
                         threshold=cfg["thresholds"][0], workdir=wd, label=name, **common)
    else:
        raise ValueError(exp)
    P.save_result(res, path)
    return {"path": path, "learner": name, "seed": seed, "status": "ran", "wall_s": time.time() - t0}


def _run_units(jobs: list, workers: int, log) -> list:
    todo = [j for j in jobs if not os.path.exists(_unit_path(j["cfg"], j["experiment"], j["learner"], j["seed"]))]
    log(f"{len(jobs)} unit(s): {len(jobs) - len(todo)} already done, {len(todo)} to run in {max(1, workers)} process(es)")
    out = []
    if workers > 1 and len(todo) > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(min(workers, len(todo))) as pool:
            for r in pool.imap_unordered(_run_unit, jobs, chunksize=1):
                log(f"  {r['learner']} seed {r['seed']}: {r['status']} ({r['wall_s']:.0f} s)")
                out.append(r)
    else:
        for j in jobs:
            r = _run_unit(j)
            log(f"  {r['learner']} seed {r['seed']}: {r['status']} ({r['wall_s']:.0f} s)")
            out.append(r)
    return out


def _load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


# ====================================================================== experiments
def _config(a) -> dict:
    from crucible.research import tasks as T
    known = T.families_by(seats=None)                     # (also loads the engines that register the families)
    train = a.train_families.split(",") if a.train_families else T.families_by("train", seats=1)
    chosen = a.eval_families.split(",") if a.eval_families else T.families_by("val", seats=1)
    second = a.second_families.split(",") if a.second_families else []
    for f in train + chosen + second:
        if f not in known:
            raise SystemExit(f"unknown mechanism family {f!r}; known: {', '.join(known)}")
    for f in train + second:
        if T.MECH_FAMILIES[f].split != "train":
            raise SystemExit(f"{f} is a {T.MECH_FAMILIES[f].split}-split family: training uses training families only")
    by_kind = {}
    for f in chosen:
        fam = T.MECH_FAMILIES[f]
        if fam.split == "test":
            raise SystemExit(f"{f} is a test-split family: test families are reached only through the sealed experiment")
        by_kind.setdefault(fam.transfer if fam.split != "train" else "within_extra", []).append(f)
    by_kind.pop("within_extra", None)
    return {"tag": a.tag, "work": os.path.abspath(a.work), "device": a.device, "opts": json.loads(a.opts),
            "train_families": train, "eval_families": {k: sorted(v) for k, v in sorted(by_kind.items())},
            "n_eval": a.n_eval, "n_envs": a.n_envs, "segments": a.segments, "budget": a.budget,
            "eval_every": a.eval_every or max(1, a.budget // 10), "thresholds": [float(x) for x in a.thresholds.split(",")],
            "sustain": a.sustain, "train_pool": a.train_pool, "second_families": second,
            "second_budget": a.second_budget, "adapt_budget": a.adapt_budget,
            "adapt_eval_every": a.adapt_eval_every or max(1, (a.adapt_budget or 1) // 5), "adapt_start": 500_000,
            "k_reset": a.k_reset}


def _finish(a, experiment: str, result: dict, log) -> dict:
    from crucible.research import protocols as P
    from crucible.research import report
    name = f"ow_benchmark_{experiment}_{a.tag}"
    result["experiment"] = {"name": experiment, "tag": a.tag, "learners": a.learners.split(","),
                            "seeds": [int(s) for s in a.seeds.split(",")], "command": " ".join(sys.argv),
                            "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    path = P.save_result(result, os.path.join(a.out, name + ".json"))
    log(f"result  {path}")
    if a.no_report:
        return {"result": path, "report": None, "figures": []}
    rep = report.render(path, os.path.join(a.figures, name), name)
    log(f"report  {rep['markdown']}  ({len(rep['figures'])} figures)")
    return {"result": path, "report": rep["markdown"], "figures": rep["figures"]}


def run(a, log=print) -> dict:
    from crucible.research import protocols as P
    cfg = _config(a)
    learners = a.learners.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    exp = a.experiment
    for n in learners:
        _factory(n, cfg)                                  # fail on an unknown name before any work is done
    if exp == "sealed":
        return _sealed(a, cfg, learners, seeds, log)
    unit_exp = "lifetime" if exp == "compare" else exp     # a comparison is made of the lifetime runs
    if a.fresh:
        shutil.rmtree(os.path.join(cfg["work"], cfg["tag"], unit_exp), ignore_errors=True)
    log(f"evaluation sets: building or loading {a.n_eval} validated task(s) per family "
        f"({', '.join(cfg['train_families'])}; {cfg['eval_families']})")
    sets = _eval_sets(cfg, a.workers)
    log("evaluation sets: " + ", ".join(f"{k}: {len(v)} tasks" for k, v in sets.items()))
    if not sets:
        raise SystemExit("no evaluation task could be built")
    if exp == "transfer":
        jobs = [{"cfg": cfg, "experiment": exp, "learner": n, "seed": "s" + "-".join(map(str, seeds)), "seeds": seeds}
                for n in learners]
    else:
        jobs = [{"cfg": cfg, "experiment": unit_exp, "learner": n, "seed": s} for n in learners for s in seeds]
    done = {(r["learner"], r["seed"]): _load(r["path"]) for r in _run_units(jobs, a.workers, log)}
    if exp == "baseline":
        res = P.combine_evaluations({n: [done[(n, s)] for s in seeds] for n in learners}, reference=a.reference,
                                    label=a.tag)
        res["seeds"] = seeds
    elif exp == "lifetime":
        runs = {n: [done[(n, s)] for s in seeds] for n in learners}
        res = {"protocol": "lifetime_set", "schema": P.SCHEMA, "label": a.tag, "runs": runs,
               "aggregate": {n: P.aggregate_lifetimes(rs) for n, rs in runs.items()}}
    elif exp == "compare":
        runs = {n: [done[(n, s)] for s in seeds] for n in learners}
        res = P.compare_runs(runs, reference=a.reference, label=a.tag)
        if a.kernel_diagnose > 0:
            _kernel_diagnoses(res, runs, sets, cfg, a.kernel_diagnose, log)
    elif exp == "timescales":
        res = {"protocol": "timescales_set", "schema": P.SCHEMA, "label": a.tag,
               "results": {n: [done[(n, s)] for s in seeds] for n in learners}}
    elif exp == "internalisation":
        res = {"protocol": "internalisation_set", "schema": P.SCHEMA, "label": a.tag,
               "results": {n: [done[(n, s)] for s in seeds] for n in learners}}
    else:
        res = {"protocol": "transfer_set", "schema": P.SCHEMA, "label": a.tag,
               "results": {n: done[(n, jobs[0]["seed"])] for n in learners}}
    res["config"] = {k: v for k, v in cfg.items() if k != "work"} if "config" not in res else res["config"]
    return _finish(a, exp, res, log)


def _kernel_diagnoses(res: dict, runs: dict, sets: dict, cfg: dict, n: int, log) -> None:
    """For every learner, replay up to `n` tasks its final checkpoint (first seed) failed, through `rig.*` tool
    calls in a kernel world, and attach what `research.diagnostics` makes of the trajectory."""
    from crucible.research import protocols as P
    for name, rs in runs.items():
        lr = _factory(name, cfg)(rs[0]["learner"]["seed"])
        ck = rs[0]["points"][-1]["checkpoint"]
        if ck and os.path.exists(ck):
            lr.load(ck)
        for set_name, ts in sets.items():
            by_id = {t.id: t for t in ts}
            failed = [r["task"] for r in rs[0]["final"][set_name]["records"] if not r["success"]][:n]
            res["diagnosis"][name][set_name]["kernel"] = P._js([P.kernel_diagnosis(lr, by_id[i], _spec()) for i in failed])
            log(f"  kernel diagnosis: {name} on {set_name}: {len(failed)} failed task(s) replayed")


def _sealed(a, cfg: dict, learners: list, seeds: list, log) -> dict:
    """The final test. The set is declared on first use; the access is made once, for every learner together."""
    from crucible.research import sealed
    if not a.ack or not a.purpose:
        raise SystemExit("the sealed experiment needs --ack (the exact acknowledgement string, see "
                         "crucible.research.sealed.SEALED_ACK) and --purpose")
    if a.sealed_set not in sealed.describe():
        d = sealed.create_set(a.sealed_set, a.sealed_families.split(",") if a.sealed_families else None,
                              n_per_family=a.sealed_n, budget=a.sealed_budget, note=a.purpose)
        log(f"sealed set declared: {d['set']}, families {d['families']}, {d['n_per_family']} per family, budget {d['budget']}")
    info = sealed.prepare(a.sealed_set, a.workers)
    log(f"sealed set {info['set']}: generation {info['generation']}, {info['n_tasks']} tasks ready, "
        f"{info['accesses']} of {info['budget']} accesses used")
    subjects, origin = {}, {}
    for n in learners:
        factory = _factory(n, cfg)
        subjects[n], origin[n] = [], []
        for s in seeds:
            lr = factory(s)
            unit = _unit_path(cfg, "lifetime", n, s)
            ck = _load(unit)["points"][-1]["checkpoint"] if os.path.exists(unit) else None
            if ck and os.path.exists(ck):
                lr.load(ck)
                origin[n].append({"seed": s, "checkpoint": ck, "trained": True})
            else:
                origin[n].append({"seed": s, "checkpoint": None, "trained": False})
            subjects[n].append(lr)
    res = sealed.evaluate(subjects, a.sealed_set, a.ack, a.purpose, reference=a.reference, workers=a.workers,
                          spec=_spec(), n_envs=cfg["n_envs"], segments=cfg["segments"], label=a.tag)
    res["subjects"] = origin
    return _finish(a, "sealed", res, log)


def _report(a, log) -> dict:
    from crucible.research import report
    out = {}
    for exp in EXPERIMENTS:
        name = f"ow_benchmark_{exp}_{a.tag}"
        path = os.path.join(a.out, name + ".json")
        if os.path.exists(path):
            out[exp] = report.render(path, os.path.join(a.figures, name), name)
            log(f"report  {out[exp]['markdown']}  ({len(out[exp]['figures'])} figures)")
    if not out:
        log(f"no result of tag {a.tag!r} under {a.out}")
    return out


def _list(log) -> dict:
    from crucible.research import learners as registry
    from crucible.research import sealed
    from crucible.research import tasks as T
    out = {"learners": {n: registry.describe(n)["kind"] for n in registry.available()},
           "families": {s: T.families_by(s, seats=1) for s in ("train", "val", "test")},
           "sealed_sets": sealed.describe()}
    log(json.dumps(out, indent=1))
    return out


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("experiment", choices=EXPERIMENTS + ("report", "list"))
    p.add_argument("--learners", default="hold,random", help="registry names, comma separated")
    p.add_argument("--opts", default="{}", help="JSON: {learner: {option: value}}")
    p.add_argument("--seeds", default="0", help="learner seeds, comma separated")
    p.add_argument("--reference", default=None, help="learner every other one is compared with (else all pairs)")
    p.add_argument("--budget", type=int, default=200_000, help="environment steps of training (source steps for transfer)")
    p.add_argument("--eval-every", type=int, default=0, help="environment steps between frozen evaluations (default budget/10)")
    p.add_argument("--adapt-budget", type=int, default=50_000, help="transfer: environment steps on each target (0: zero-shot only)")
    p.add_argument("--adapt-eval-every", type=int, default=0)
    p.add_argument("--second-families", default="", help="internalisation: families of the second training (retention)")
    p.add_argument("--second-budget", type=int, default=0)
    p.add_argument("--thresholds", default="0.5", help="score thresholds for steps-to-threshold")
    p.add_argument("--sustain", type=int, default=2)
    p.add_argument("--train-families", default="", help="default: every one-seat training family")
    p.add_argument("--eval-families", default="", help="validation families for the transfer sets (default: all one-seat)")
    p.add_argument("--n-eval", type=int, default=8, help="validated tasks per family in each evaluation set")
    p.add_argument("--train-pool", type=int, default=64, help="training tasks per family, cycled (0: endless stream)")
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--segments", type=int, default=4)
    p.add_argument("--workers", type=int, default=1, help="processes for units and for building tasks")
    p.add_argument("--device", default="cpu")
    p.add_argument("--tag", default="default", help="names the result files and the working directory")
    p.add_argument("--out", default=os.path.join(ROOT, "docs", "experiments"))
    p.add_argument("--figures", default=os.path.join(ROOT, "docs", "figures"))
    p.add_argument("--work", default=os.path.join(os.environ.get("CRUCIBLE_CACHE") or
                                                  os.path.join(os.path.expanduser("~"), ".cache", "crucible"), "ow_benchmark"))
    p.add_argument("--fresh", action="store_true", help="discard this experiment's units and checkpoints first")
    p.add_argument("--no-report", action="store_true", help="write the result only; render later with `report`")
    p.add_argument("--kernel-diagnose", type=int, default=0,
                   help="compare: failed tasks per learner and set to replay through tool calls in a kernel world")
    p.add_argument("--k-reset", type=int, default=1, help="timescales: ticks between context resets in the ablated copy")
    p.add_argument("--sealed-set", default="ow-final", help="sealed: name of the set")
    p.add_argument("--sealed-families", default="", help="sealed: test-split families (default: all one-seat)")
    p.add_argument("--sealed-n", type=int, default=8)
    p.add_argument("--sealed-budget", type=int, default=1)
    p.add_argument("--purpose", default="", help="sealed: why the set is opened (logged)")
    p.add_argument("--ack", default="", help="sealed: the exact acknowledgement string")
    return p


def main(argv=None, log=print) -> dict:
    a = parser().parse_args(argv)
    if a.experiment == "list":
        return _list(log)
    if a.experiment == "report":
        return _report(a, log)
    return run(a, log)


if __name__ == "__main__":         # required: worker processes re-import this file
    main()
