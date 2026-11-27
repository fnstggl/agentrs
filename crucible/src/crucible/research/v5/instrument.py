"""The research instrument: protocols that tell apart what a learner did (MASTER_SPEC Parts 15, 15.4, 15.6; U.4,
SP-15.0, SP-15.6).

Every learner goes through the same battery, with several independent learner seeds on the same worlds (crossed
design, `hstats.crossed_diff`). All scores are verified, normalised outcomes (`episodes.score_episode`); learners
see only their own observations (``observation_only``).

    untrained     scores on unfamiliar worlds (new seeds of the training families) before any training: episode
                  score, blind score (no feedback within the episode), early and late rates
    training      the same stream for every learner: `train_tasks` training worlds, each played `repeats` times
    repeat probe  in-process frozen scores on worlds seen in training, before and after training
    in process    frozen scores on the unfamiliar worlds after training, with everything the learner keeps
    weights only  the same worlds scored by `fresh.weights_only_eval` from the checkpoint alone
    transfer      practice curves (`curves.practice_curve`) on held-out families, trained versus untrained copies
                  (the trained copy starts with its memory cleared: only what it learned elsewhere is kept)

Explanations (Part 15 introduction), each a prespecified effect on the normalised scale (`EFFECTS`) whose crossed
interval must exclude zero:

    pretrained_competence    score before any training, with no within-episode feedback (blind episode), above
                             the hold baseline's on the same worlds: competent without adapting
    context_adaptation       score minus blind score before training: what feedback within the episode
                             contributed (the blind episode is the learner's own no-feedback ablation)
    memorisation             gain on seen worlds exceeds the gain on unfamiliar worlds
    persistent_state         in-process gain that the weights-only evaluation does not reproduce
    weight_learning          weights-only gain on unfamiliar worlds, with changed parameters
    meta_learning            within-episode contribution (score - blind score) after training, weights only,
                             above the same contribution before training: it learned to adapt better
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
    # the transfer control keeps a per-task search state (its memory, by construction), so it memorises the worlds
    # it practised; that is part of its known capability, not an error of the instrument (corrected after the
    # 2026-11-26 development run, which labelled it so)
    "transfer": (("transferable_efficiency",), ("persistent_state", "pretrained_competence", "context_adaptation")),
    # the sham is the transfer control credited with noise: it keeps the same per-task search states, so it too
    # retains world-specific settings (a random walk from the published defaults, clipped at the range ends, drifts
    # inward); whether that helps on seen worlds is a property of the worlds' geometry, not something it learned, so
    # no claim about memorisation was ever established for it - it is the known negative for transfer (corrected
    # after the 2026-11-26 wave-3b development battery, seed 0: seen-world advantage 0.19, interval low -0.13)
    "transfer_sham": ((), ("transferable_efficiency", "persistent_state", "pretrained_competence")),
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
    repeats: int = 3
    # ten unfamiliar worlds per family, every training world probed again, four held-out tasks per transfer family:
    # with 4 / 12 / 2 the crossed intervals of real effects (0.13 - 0.49) still reached below zero (2026-11-26 run)
    eval_seeds: tuple[int, ...] = tuple(range(1_000_101, 1_000_111))
    transfer_seeds: tuple[int, ...] = (1_000_201, 1_000_202, 1_000_203, 1_000_204)
    repeat_probe: int = 24
    transfer_budgets: tuple[int, ...] = (0, 1, 2, 4, 8, 16)
    q: float = 0.3
    pretrain_tasks: int = 48
    world_seed0: int = 300

    def to_json(self) -> dict:
        return asdict(self)


ESTABLISH_SEEDS = (1_500_001, 1_500_002, 1_500_003)
ESTABLISH_MIN = 0.05


def _cache_path(name: str, key) -> str:
    import hashlib
    import json
    base = os.environ.get("CRUCIBLE_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "crucible")
    from .distribution import source_digest
    h = hashlib.sha256(json.dumps([source_digest(), key], sort_keys=True, default=str).encode()).hexdigest()[:20]
    d = os.path.join(base, "research_v5")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{name}-{h}.json")


def establish(registry: Registry | None = None, families: Sequence[str] | None = None) -> dict:
    """Independently establish, per family, the capabilities the positive controls are built to have, on seeds
    no protocol uses (`ESTABLISH_SEEDS`, validation partition): the within-episode adapter's gain over its own blind
    run, and the weights learner's gain on new seeds after training on that family alone. Cached by source digest."""
    import json
    reg = registry or build_registry()
    fams = list(families or lever_entries(reg, "train"))
    path = _cache_path("establish", fams)
    if os.path.exists(path):
        return json.load(open(path))
    out = {}
    for f in fams:
        rng = np.random.default_rng(5)
        ad = LR.make("adapter", 0)
        ctx = [score_episode(reg, f, s, ad, learn=False, rng=rng)["score"] -
               score_episode(reg, f, s, ad, learn=False, rng=rng, blind=True)["score"] for s in ESTABLISH_SEEDS]
        w = LR.make("weights", 0)
        base = [score_episode(reg, f, s, w, learn=False, rng=rng)["score"] for s in ESTABLISH_SEEDS]
        for i in range(8):
            for _ in range(3):
                run_episode(build_env(reg[f], 700_000 + i, {}, {"mode": "assigned"}), w, learn=True, rng=rng,
                            verify=False)
        w.frozen = True
        after = [score_episode(reg, f, s, w, learn=False, rng=rng)["score"] for s in ESTABLISH_SEEDS]
        out[f] = {"adapter_gain": float(np.mean(ctx)), "weights_gain": float(np.mean(after) - np.mean(base)),
                  "adapter_gains": ctx, "weights_gains": [a - b for a, b in zip(after, base)]}
    json.dump(out, open(path, "w"))
    return out


def establish_transfer(registry: Registry, train: Sequence[str], candidates: Sequence[str],
                       budgets=(0, 1, 2, 4, 8, 16)) -> dict:
    """Per candidate held-out family: AUC of a transfer learner trained on `train` minus an untrained one, on the
    establishment seeds (independent of every protocol seed). Cached."""
    import json
    path = _cache_path("establish_transfer", [list(train), list(candidates), list(budgets)])
    if os.path.exists(path):
        return json.load(open(path))
    rng = np.random.default_rng(9)
    T = LR.make("transfer", 999)
    for i in range(24):
        f = train[i % len(train)]
        for _ in range(4):
            run_episode(build_env(registry[f], 800_000 + i, {}, {"mode": "assigned"}), T, learn=True, rng=rng,
                        verify=False)
    out = {}
    for f in candidates:
        g = []
        for s in ESTABLISH_SEEDS:
            T.clear_memory()
            a = CV.practice_curve(registry, {"entry": f, "seed": s}, T, budgets=budgets, rng=rng)["auc"]
            u = CV.practice_curve(registry, {"entry": f, "seed": s}, LR.make("transfer", 998), budgets=budgets,
                                  rng=rng)["auc"]
            g.append(a - u)
        out[f] = {"transfer_gain": float(np.mean(g)), "gains": g}
    json.dump(out, open(path, "w"))
    return out


def public_face(registry: Registry, family: str) -> np.ndarray:
    """The brief vector a learner is given for `family` (episodes.brief_vector of its public text, establishment seed)."""
    from .episodes import LeverInterface
    env = build_env(registry[family], ESTABLISH_SEEDS[0], {}, {"mode": "assigned"})
    return np.asarray(LeverInterface(env, env.agents()[0]).context()["brief_vec"], dtype=float)


def establish_mixed(registry: Registry, families: Sequence[str], *, train_tasks: int = 24, repeats: int = 3,
                    learner_seeds: Sequence[int] = (0, 1)) -> dict:
    """The weights learner's capability in the battery's own regime: trained on `families` interleaved
    (`train_tasks` worlds on establishment-only training seeds, `repeats` episodes each), then scored frozen on the
    establishment seeds of every family (no protocol uses those seeds). Per family: mean gain over learner seeds
    and establishment seeds. Single-family establishment alone overstated it: families whose public faces are
    nearly identical but whose optima differ cannot all be learned by one network (2026-11-26 diagnosis). Cached."""
    import json
    fams = list(families)
    path = _cache_path("establish_mixed", [fams, train_tasks, repeats, list(learner_seeds)])
    if os.path.exists(path):
        return json.load(open(path))
    gains: dict[str, list[float]] = {f: [] for f in fams}
    for ls in learner_seeds:
        rng = np.random.default_rng(31 + ls)
        w = LR.make("weights", 50 + ls)
        base = {f: [score_episode(registry, f, s, w, learn=False, rng=rng)["score"] for s in ESTABLISH_SEEDS]
                for f in fams}
        for i in range(train_tasks):
            f = fams[i % len(fams)]
            for _ in range(repeats):
                run_episode(build_env(registry[f], 710_000 + 1000 * ls + i, {}, {"mode": "assigned"}), w, learn=True,
                            rng=rng, verify=False)
        w.frozen = True
        for f in fams:
            after = [score_episode(registry, f, s, w, learn=False, rng=rng)["score"] for s in ESTABLISH_SEEDS]
            gains[f] += [a - b for a, b in zip(after, base[f])]
    out = {f: {"weights_gain": float(np.mean(g)), "gains": g} for f, g in gains.items()}
    json.dump(out, open(path, "w"))
    return out


FACE_DISTINCT = 0.9          # maximum cosine between the public faces (brief vectors) of two training families


def default_design(registry: Registry | None = None, *, n_train: int = 6, n_transfer: int = 4, **kw) -> Design:
    """Families on which the controls' capabilities were established independently, on seeds no protocol uses:

    1. single-family establishment (`establish`): the adapter's and the weights learner's gains, each at least
       `ESTABLISH_MIN`; candidates ranked by the smaller of the two;
    2. public faces distinct (`FACE_DISTINCT`): a candidate whose brief vector is nearly that of a family already
       taken is skipped - a learner conditioned on what it is shown cannot tell such families apart;
    3. mixed-regime establishment (`establish_mixed`): the chosen set is trained together as in the battery; a
       family whose established gain there is below `ESTABLISH_MIN` is replaced by the next candidate (at most
       three rounds).

    Transfer families are the `n_transfer` other families with the largest established transfer gain
    (`establish_transfer`, at least `ESTABLISH_MIN`). All measured items are validation-partition items."""
    reg = registry or build_registry()
    est = establish(reg)
    ranked = sorted((f for f, v in est.items() if v["adapter_gain"] >= ESTABLISH_MIN and
                     v["weights_gain"] >= ESTABLISH_MIN), key=lambda f: -min(est[f]["adapter_gain"],
                                                                             est[f]["weights_gain"]))
    faces = {f: public_face(reg, f) for f in ranked}

    def cos(a, b):
        na, nb = np.linalg.norm(faces[a]), np.linalg.norm(faces[b])
        return float(faces[a] @ faces[b] / (na * nb)) if na > 0 and nb > 0 else 0.0
    distinct: list[str] = []
    for f in ranked:
        if all(cos(f, g) < FACE_DISTINCT for g in distinct):
            distinct.append(f)
    train = distinct[:n_train]
    pool = distinct[n_train:]
    for _ in range(3):
        mixed = establish_mixed(reg, sorted(train))
        bad = [f for f in train if mixed[f]["weights_gain"] < ESTABLISH_MIN]
        if not bad or not pool:
            break
        train = [f for f in train if f not in bad] + pool[:len(bad)]
        pool = pool[len(bad):]
    train = tuple(sorted(train))
    rest = [f for f in est if f not in train]
    tr = establish_transfer(reg, train, rest)
    tf = sorted((f for f in rest if tr[f]["transfer_gain"] >= ESTABLISH_MIN), key=lambda f: -tr[f]["transfer_gain"])
    return Design(train_families=train, eval_families=train, transfer_families=tuple(sorted(tf[:n_transfer])), **kw)


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


def _scores(reg, learner, worlds, rng, blind: bool = True) -> np.ndarray:
    """[world, (score, blind score, early rate, late rate)] of frozen episodes; the blind episode gives the learner
    no consequences (0 signal), so score - blind score is what within-episode adaptation contributed."""
    was = learner.frozen
    learner.frozen = True
    out = []
    for f, s in worlds:
        r = score_episode(reg, f, s, learner, learn=False, rng=rng)
        b = score_episode(reg, f, s, learner, learn=False, rng=rng, blind=True)["score"] if blind else r["score"]
        out.append((r["score"], b, r["early_rate"], r["late_rate"]))
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
    rb = _scores(reg, L, WR, rng, blind=False)
    # transfer: untrained copy
    tu = []
    for f, s in WT:
        c = _make(arch, seed, d, reg)
        tu.append(CV.practice_curve(reg, {"entry": f, "seed": s}, c, budgets=d.transfer_budgets, rng=rng, q=d.q))
    for f, s in T:
        for _ in range(d.repeats):
            run_episode(build_env(reg[f], s, {}, {"mode": "assigned"}), L, learn=True, rng=rng)
    h1 = LR.params_hash(L.parameters())
    ra = _scores(reg, L, WR, rng, blind=False)
    ip = _scores(reg, L, W1, rng)
    man = FR.save_checkpoint(L, os.path.join(workdir, f"{arch}-{seed}"))
    fo = FR.weights_only_eval(man, [{"entry": f, "seed": s} for f, s in W1])
    if not fo.get("ok"):
        raise RuntimeError(f"weights-only evaluation failed for {arch}/{seed}: {fo.get('refused') or fo.get('error')}")
    wf = np.array([(x["score"], x["blind_score"], x["early_rate"], x["late_rate"]) for x in fo["scores"]])
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


# Labels are decided at a Bonferroni level across the seven explanations, so the family-wise rate of a false label
# per control is at most 5%: at the plain 95% level a noisy null explanation got a label in about 2.5% of simulated
# batteries (5 seeds x 24-50 worlds, seed sd 0.1-0.2, world sd 0.5-0.6), i.e. a run with ~8 noisy null pairs would
# mislabel some control about one time in six (decided before the wave-3b confirmation run)
LABEL_LEVEL = 1.0 - 0.05 / len(EXPLANATIONS)


def _one(X: np.ndarray) -> dict:
    """Crossed one-sample test of a [seed, world] matrix of differences against 0 (interval at `LABEL_LEVEL`)."""
    return HS.crossed_diff(np.zeros_like(X), X, level=LABEL_LEVEL)


def _label(res: Mapping, effect: float) -> bool:
    return bool(res["lo"] > 0 and res["estimate"] >= effect and min(res["seeds"]) >= HS.MIN_SEEDS)


def classify(runs: Sequence[Mapping], hold: np.ndarray, d: Design) -> dict:
    """Labels found for one learner (all its seeds), with the statistics behind each."""
    A = lambda k, i: np.array([np.array(r[k])[:, i] for r in runs])        # noqa: E731
    hs = hold[:, 0]
    stats = {}
    stats["pretrained_competence"] = _one(A("untrained", 1) - hs[None, :])
    stats["context_adaptation"] = _one(A("untrained", 0) - A("untrained", 1))
    rep_gain = A("rep_after", 0) - A("rep_before", 0)
    new_gain = A("in_process", 0) - A("untrained", 0)
    rg, ng = _one(rep_gain), _one(new_gain)
    est = rg["estimate"] - ng["estimate"]
    se = float(np.hypot(rg["se"], ng["se"]))
    df = min(rg["df"], ng["df"])
    q = HS._t_quantile(1 - (1 - LABEL_LEVEL) / 2, df)
    stats["memorisation"] = {"estimate": est, "se": se, "lo": est - q * se, "hi": est + q * se, "df": df,
                             "seeds": rg["seeds"], "rep_gain": rg, "new_gain": ng}
    stats["persistent_state"] = _one(A("in_process", 0) - A("weights_only", 0))
    wl = _one(A("weights_only", 0) - A("untrained", 0))
    stats["weight_learning"] = dict(wl, params_changed=all(r["params_changed"] for r in runs))
    stats["meta_learning"] = _one((A("weights_only", 0) - A("weights_only", 1)) - (A("untrained", 0) -
                                                                                   A("untrained", 1)))
    Tu = np.array([[c["auc"] for c in r["transfer_untrained"]] for r in runs])
    Tt = np.array([[c["auc"] for c in r["transfer_trained"]] for r in runs])
    tr = HS.crossed_diff(Tu, Tt, paired_seeds=True, level=LABEL_LEVEL)
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
        c["fresh_checks"] = {k: all(r["fresh_checks"].get(k) for r in runs) for k in runs[0]["fresh_checks"]}
        c["params_changed"] = [bool(r["params_changed"]) for r in runs]
        c["seeds"] = [r["seed"] for r in runs]
        out[a] = c
    return {"design": d.to_json(), "controls": out, "effects": dict(EFFECTS)}


# Confirmation seeds (wave 3b repair, D-69 addendum 2): the development batteries (learner seeds 0 and 1 on the
# default seeds) informed the transfer_sham specification and LABEL_LEVEL, so the confirmation run uses learner seeds,
# training-world seeds, unfamiliar-world seeds and transfer-task seeds that no development run used. Family selection
# (`default_design`) uses only the establishment seeds, which no battery uses.
CONFIRMATION = {"seeds": (10, 11, 12, 13, 14), "world_seed0": 3300,
                "eval_seeds": tuple(range(1_003_101, 1_003_111)),
                "transfer_seeds": (1_003_201, 1_003_202, 1_003_203, 1_003_204)}


def controls_result(archs: Sequence[str] = tuple(CONTROLS), *, seeds: Sequence[int] | None = None,
                    jobs: int | None = None, confirmation: bool = True) -> dict:
    """`run_controls` on the default design (established families) at the confirmation seeds (`CONFIRMATION`;
    `confirmation=False`: the development seeds), cached on disk by source digest, design and controls, so the gate
    files that read the same confirmation run (U.3, U.4, SP-15.0, SP-15.4, SP-15.6) share one execution. `jobs`
    defaults to $CRUCIBLE_RESEARCH_JOBS (run through the slot scheduler)."""
    import json
    kw = dict(CONFIRMATION) if confirmation else {}
    if seeds is not None:
        kw["seeds"] = tuple(seeds)
    d = default_design(**{k: tuple(v) if isinstance(v, (list, tuple)) else v for k, v in kw.items()})
    path = _cache_path("controls", [d.to_json(), list(archs)])
    if os.path.exists(path):
        res = json.load(open(path))
        res["cached"] = path
        return res
    jobs = int(jobs if jobs is not None else os.environ.get("CRUCIBLE_RESEARCH_JOBS", "1"))
    res = run_controls(list(archs), d, jobs=jobs)
    json.dump(res, open(path, "w"), default=_json_default)
    res["cached"] = None
    return res


def _json_default(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x).__name__)


def transfer_curves(*, seeds: Sequence[int] = (0, 1, 2), budgets: Sequence[int] = CV.SCHEDULE + CV.LONG,
                    arch: str = "transfer") -> dict:
    """Learning curves of untrained and trained copies of `arch` on the design's held-out (transfer) families, per
    learner seed and task, with N_q at the design's threshold, retention (score on the first task after practising
    the others) and accounting. Cached by source digest."""
    import json
    reg = build_registry()
    d = default_design(reg)
    path = _cache_path("transfer_curves", [d.to_json(), list(seeds), list(budgets), arch])
    if os.path.exists(path):
        return json.load(open(path))
    tasks = [{"entry": f, "seed": 1_000_211} for f in d.transfer_families]
    runs = {"untrained": [], "trained": []}
    retention = []
    for sd in seeds:
        rng = np.random.default_rng(4000 + sd)
        T = LR.make(arch, sd)
        for i in range(d.train_tasks):
            f = d.train_families[i % len(d.train_families)]
            for _ in range(4):
                run_episode(build_env(reg[f], 740_000 + 100 * sd + i, {}, {"mode": "assigned"}), T, learn=True,
                            rng=rng, verify=False)
        T.clear_memory()
        first = None
        for t in tasks:
            r = CV.practice_curve(reg, t, T, budgets=budgets, rng=rng, q=d.q)
            runs["trained"].append(dict(r, task=f"{t['entry']}:{t['seed']}", learner_seed=sd))
            if first is None:
                first = (t, r["curve"][-1]["score"])
        s_after, _ = CV._score(reg, first[0], T, rng)        # retention of the first task after the others
        retention.append({"learner_seed": sd, "task": first[0]["entry"], "score_after_practice": first[1],
                          "score_after_other_tasks": s_after})
        for t in tasks:
            U = LR.make(arch, sd)
            r = CV.practice_curve(reg, t, U, budgets=budgets, rng=rng, q=d.q)
            runs["untrained"].append(dict(r, task=f"{t['entry']}:{t['seed']}", learner_seed=sd))
    out = {"design": d.to_json(), "q": d.q, "budgets": list(budgets), "runs": runs, "retention": retention,
           "acceleration": CV.acceleration(runs["untrained"], runs["trained"], n_boot=500)}
    json.dump(out, open(path, "w"), default=_json_default)
    return out
