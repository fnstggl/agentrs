"""The research instrument: protocols that tell apart what a learner did (MASTER_SPEC Parts 15, 15.4, 15.6; U.4,
SP-15.0, SP-15.6).

Every learner goes through the same battery, with several independent learner seeds on the same worlds (crossed
design, `hstats.crossed_diff`). All scores are verified, normalised outcomes (`episodes.score_episode`); learners
see only their own observations (``observation_only``).

    untrained     scores on unfamiliar worlds (new seeds of the training families) before any training: episode
                  score, early rate (first quarter) and late rate (last quarter)
    training      the same stream for every learner: `train_tasks` training worlds, each played `repeats` times
    repeat probe  in-process frozen scores on worlds seen in training, before and after training
    in process    frozen scores on the unfamiliar worlds after training, with everything the learner keeps
    weights only  the same worlds scored by `fresh.weights_only_eval` from the checkpoint alone
    transfer      practice curves (`curves.practice_curve`) on held-out families, trained versus untrained copies
                  (the trained copy starts with its memory cleared: only what it learned elsewhere is kept)

Explanations (Part 15 introduction), each a prespecified effect on the normalised scale (`EFFECTS`) whose crossed
interval must exclude zero:

    pretrained_competence    early rate before any adaptation above the hold baseline's on the same worlds
    context_adaptation       within-episode improvement (late - early) above the hold baseline's, before training
    memorisation             gain on seen worlds exceeds the gain on unfamiliar worlds
    persistent_state         in-process gain that the weights-only evaluation does not reproduce
    weight_learning          weights-only gain on unfamiliar worlds, with changed parameters
    meta_learning            within-episode improvement after training (weights only) above before training
    transferable_efficiency  area under the practice curve on held-out families, trained above untrained (with
                             N_q and a censoring-aware acceleration reported)

`CONTROLS` states each control's known capability as labels that must be found and labels that must not be found;
`classify` returns what the instrument found, and `check` compares.
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

import numpy as np

from . import curves as CV
from . import fresh as FR
from . import hstats as HS
from . import learners as LR
from .distribution import Registry, build_registry
from .episodes import lever_entries, run_episode, score_episode
from .sampler import build_env

EXPLANATIONS = ("memorisation", "pretrained_competence", "context_adaptation", "persistent_state",
                "weight_learning", "meta_learning", "transferable_efficiency")
EFFECTS = {"pretrained_competence": 0.05, "context_adaptation": 0.10, "memorisation": 0.05, "persistent_state": 0.03,
           "weight_learning": 0.03, "meta_learning": 0.10, "transferable_efficiency": 0.05}
# known capabilities: (labels that must be found, labels that must not be found)
CONTROLS = {
    "hold": ((), EXPLANATIONS),
    "adapter": (("context_adaptation",), ("memorisation", "persistent_state", "weight_learning",
                                          "pretrained_competence", "meta_learning")),
    "memoriser": (("memorisation",), ("weight_learning", "persistent_state", "pretrained_competence",
                                      "context_adaptation", "meta_learning")),
    "weights": (("weight_learning",), ("memorisation", "persistent_state", "context_adaptation",
                                       "pretrained_competence", "meta_learning")),
    "stateful": (("persistent_state",), ("weight_learning", "memorisation", "pretrained_competence",
                                         "meta_learning")),
    "pretrained": (("pretrained_competence",), ("weight_learning", "persistent_state", "memorisation",
                                                "context_adaptation", "meta_learning", "transferable_efficiency")),
    "meta": (("meta_learning",), ("memorisation", "persistent_state", "pretrained_competence")),
    "transfer": (("transferable_efficiency",), ("memorisation", "persistent_state", "pretrained_competence",
                                                "context_adaptation")),
    "transfer_sham": ((), ("transferable_efficiency", "memorisation", "persistent_state", "pretrained_competence")),
}
KIND_OF = {"hold": "cannot_learn", "adapter": "within_episode_only", "memoriser": "memorises_trajectories",
           "weights": "updates_weights", "transfer": "genuine_transfer"}


@dataclass
class Design:
    train_families: tuple[str, ...]
    eval_families: tuple[str, ...]
    transfer_families: tuple[str, ...]
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    train_tasks: int = 24
    repeats: int = 4
    eval_seeds: tuple[int, ...] = (1_000_101, 1_000_102)
    transfer_seeds: tuple[int, ...] = (1_000_201,)
    repeat_probe: int = 8
    transfer_budgets: tuple[int, ...] = (0, 1, 2, 4, 8, 16)
    q: float = 0.3
    pretrain_tasks: int = 48
    world_seed0: int = 300

    def to_json(self) -> dict:
        return asdict(self)


def default_design(registry: Registry | None = None, *, n_eval: int = 8, n_transfer: int = 4, **kw) -> Design:
    """Training families: rapid primitive-graph lever families of the training split; unfamiliar worlds: new seeds
    (validation partition) of a fixed subset of them; transfer: training-split families never used for training
    (unfamiliar to the learner) - all validation items, never evaluation or sealed ones."""
    reg = registry or build_registry()
    fams = lever_entries(reg, "train")
    rng = np.random.default_rng(20261126)
    order = [fams[i] for i in rng.permutation(len(fams))]
    transfer = tuple(sorted(order[:n_transfer]))
    train = tuple(sorted(order[n_transfer:]))
    ev = tuple(sorted(train[:: max(1, len(train) // n_eval)][:n_eval]))
    return Design(train_families=train, eval_families=ev, transfer_families=transfer, **kw)


def _training_worlds(d: Design) -> list[tuple[str, int]]:
    return [(d.train_families[i % len(d.train_families)], d.world_seed0 + i) for i in range(d.train_tasks)]


def _make(arch: str, seed: int, d: Design, reg: Registry):
    if arch == "transfer_sham":
        return LR.make("transfer", seed, sham=True)
    if arch == "pretrained":                       # competence established by pretraining a weights network
        w = LR.make("weights", seed)
        rng = np.random.default_rng(seed + 991)
        for i in range(d.pretrain_tasks):
            f, s = d.train_families[i % len(d.train_families)], 600_000 + 1000 * seed + i
            for _ in range(d.repeats):
                run_episode(build_env(reg[f], s, {}, {"mode": "assigned"}), w, learn=True, rng=rng)
        p = LR.make("pretrained", seed)
        p.load_parameters(w.parameters())
        return p
    return LR.make(arch, seed)


def _scores(reg, learner, worlds, rng) -> np.ndarray:
    was = learner.frozen
    learner.frozen = True
    out = []
    for f, s in worlds:
        r = score_episode(reg, f, s, learner, learn=False, rng=rng)
        out.append((r["score"], r["early_rate"], r["late_rate"]))
    learner.frozen = was
    return np.array(out)


def battery(arch: str, seed: int, d: Design, workdir: str) -> dict:
    """All measurements of one learner seed (module doc)."""
    reg = build_registry()
    rng = np.random.default_rng(10_000 + seed)
    L = _make(arch, seed, d, reg)
    W1 = [(f, s) for f in d.eval_families for s in d.eval_seeds]
    T = _training_worlds(d)
    WR = T[: d.repeat_probe]
    WT = [(f, s) for f in d.transfer_families for s in d.transfer_seeds]
    h0 = LR.params_hash(L.parameters())
    u = _scores(reg, L, W1, rng)
    rb = _scores(reg, L, WR, rng)
    # transfer: untrained copy
    tu = []
    for f, s in WT:
        c = _make(arch, seed, d, reg)
        tu.append(CV.practice_curve(reg, {"entry": f, "seed": s}, c, budgets=d.transfer_budgets, rng=rng, q=d.q))
    for f, s in T:
        for _ in range(d.repeats):
            run_episode(build_env(reg[f], s, {}, {"mode": "assigned"}), L, learn=True, rng=rng)
    h1 = LR.params_hash(L.parameters())
    ra = _scores(reg, L, WR, rng)
    ip = _scores(reg, L, W1, rng)
    man = FR.save_checkpoint(L, os.path.join(workdir, f"{arch}-{seed}"))
    fo = FR.weights_only_eval(man, [{"entry": f, "seed": s} for f, s in W1])
    if not fo.get("ok"):
        raise RuntimeError(f"weights-only evaluation failed for {arch}/{seed}: {fo.get('refused') or fo.get('error')}")
    wf = np.array([(x["score"], x["early_rate"], x["late_rate"]) for x in fo["scores"]])
    # transfer: trained copy, memory cleared (what it learned elsewhere is in its parameters)
    tt = []
    for f, s in WT:
        L.clear_memory()
        L.reset_state()
        tt.append(CV.practice_curve(reg, {"entry": f, "seed": s}, L, budgets=d.transfer_budgets, rng=rng, q=d.q))
    return {"arch": arch, "seed": seed, "params_changed": h0 != h1, "untrained": u.tolist(), "rep_before": rb.tolist(),
            "rep_after": ra.tolist(), "in_process": ip.tolist(), "weights_only": wf.tolist(),
            "fresh_checks": fo["checks"], "transfer_untrained": tu, "transfer_trained": tt,
            "worlds": [list(w) for w in W1], "repeat_worlds": [list(w) for w in WR],
            "transfer_tasks": [list(w) for w in WT]}


def hold_baseline(d: Design) -> np.ndarray:
    reg = build_registry()
    W1 = [(f, s) for f in d.eval_families for s in d.eval_seeds]
    return _scores(reg, LR.make("hold", 0), W1, np.random.default_rng(0))


def _one(X: np.ndarray) -> dict:
    """Crossed one-sample test of a [seed, world] matrix of differences against 0."""
    return HS.crossed_diff(np.zeros_like(X), X)


def _label(res: Mapping, effect: float) -> bool:
    return bool(res["lo"] > 0 and res["estimate"] >= effect and min(res["seeds"]) >= HS.MIN_SEEDS)


def classify(runs: Sequence[Mapping], hold: np.ndarray, d: Design) -> dict:
    """Labels found for one learner (all its seeds), with the statistics behind each."""
    A = lambda k, i: np.array([np.array(r[k])[:, i] for r in runs])        # noqa: E731
    he, hl = hold[:, 1], hold[:, 2]
    stats = {}
    stats["pretrained_competence"] = _one(A("untrained", 1) - he[None, :])
    stats["context_adaptation"] = _one((A("untrained", 2) - A("untrained", 1)) - (hl - he)[None, :])
    rep_gain = A("rep_after", 0) - A("rep_before", 0)
    new_gain = A("in_process", 0) - A("untrained", 0)
    rg, ng = _one(rep_gain), _one(new_gain)
    est = rg["estimate"] - ng["estimate"]
    se = float(np.hypot(rg["se"], ng["se"]))
    df = min(rg["df"], ng["df"])
    q = HS._t_quantile(0.975, df)
    stats["memorisation"] = {"estimate": est, "se": se, "lo": est - q * se, "hi": est + q * se, "df": df,
                             "seeds": rg["seeds"], "rep_gain": rg, "new_gain": ng}
    stats["persistent_state"] = _one(A("in_process", 0) - A("weights_only", 0))
    wl = _one(A("weights_only", 0) - A("untrained", 0))
    stats["weight_learning"] = dict(wl, params_changed=all(r["params_changed"] for r in runs))
    stats["meta_learning"] = _one((A("weights_only", 2) - A("weights_only", 1)) - (A("untrained", 2) -
                                                                                   A("untrained", 1)))
    Tu = np.array([[c["auc"] for c in r["transfer_untrained"]] for r in runs])
    Tt = np.array([[c["auc"] for c in r["transfer_trained"]] for r in runs])
    tr = HS.crossed_diff(Tu, Tt, paired_seeds=True)
    unt = [dict(c, task=f"{c['entry']}:{c['seed']}", learner_seed=r["seed"]) for r in runs
           for c in r["transfer_untrained"]]
    exp = [dict(c, task=f"{c['entry']}:{c['seed']}", learner_seed=r["seed"]) for r in runs
           for c in r["transfer_trained"]]
    tr["acceleration"] = CV.acceleration(unt, exp, n_boot=300)
    stats["transferable_efficiency"] = tr
    labels = []
    for k in EXPLANATIONS:
        ok = _label(stats[k], EFFECTS[k])
        if k == "weight_learning":
            ok = ok and stats[k]["params_changed"]
        if ok:
            labels.append(k)
    return {"labels": labels, "stats": stats}


def check(arch: str, labels: Sequence[str]) -> dict:
    must, must_not = CONTROLS[arch]
    missing = [x for x in must if x not in labels]
    wrong = [x for x in must_not if x in labels]
    return {"correct": not missing and not wrong, "missing": missing, "unexpected": wrong}


def _battery_job(args):
    arch, seed, d_json, workdir = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return battery(arch, seed, Design(**{k: tuple(v) if isinstance(v, list) else v for k, v in d_json.items()}),
                   workdir)


def run_controls(archs: Sequence[str], d: Design, *, jobs: int = 1, workdir: str | None = None) -> dict:
    """Battery and classification for every control in `archs` (all seeds of `d`)."""
    workdir = workdir or tempfile.mkdtemp(prefix="crucible-v5-controls-")
    tasks = [(a, s, d.to_json(), workdir) for a in archs for s in d.seeds]
    if jobs > 1:
        import multiprocessing as mp
        with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn")) as ex:
            res = list(ex.map(_battery_job, tasks))
    else:
        res = [_battery_job(t) for t in tasks]
    hold = hold_baseline(d)
    out = {}
    for a in archs:
        runs = [r for r in res if r["arch"] == a]
        c = classify(runs, hold, d)
        c["check"] = check(a, c["labels"]) if a in CONTROLS else None
        c["fresh_checks_all_passed"] = all(all(r["fresh_checks"].values()) for r in runs)
        out[a] = c
    return {"design": d.to_json(), "controls": out, "effects": dict(EFFECTS)}
