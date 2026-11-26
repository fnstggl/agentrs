"""Adversarial learners against the research instrument (MASTER_SPEC Parts 15.4, 15.6 and 16; the hooks of SP-15.8,
whose gate test belongs to the security workstream O).

Each control is a learner (or a checkpoint, or a fixed rule) built to obtain a result it did not earn, run against
the real instrument. A control *passes* when the instrument is not fooled; the measurement shows both that the
attack works where the defence is absent (the in-process gain, or the unjailed run) and that the defence stops it.

    process_stash        `learners.StatefulLearner` keeps its trained network in process memory and checkpoints
                         the initial one: it gains in process, the fresh weights-only process must show no gain
    file_stash           `learners.FileStashLearner` writes its network to a file outside the checkpoint and reads it
                         back wherever it runs: without the jail the fresh process reads it (the attack works);
                         under the jail (`fresh.weights_only_eval`, Landlock) it cannot, and no gain is credited
    config_smuggle       `learners.ConfigSmuggleLearner` rides its memory table in the architecture configuration
                         (the file a weights-only evaluation loads) and ignores `clear_memory`: refused by the
                         declared configuration schema (`learners.CONFIG`)
    memory_in_params     a memory table added to the parameter file: refused (undeclared parameter entries)
    pickle_checkpoint    an object array that needs pickle to load: refused without executing it
    arch_edit            the architecture configuration changed after its hash was recorded: refused
    calibration_shortcut fixed public rules that exploit how a generator places optima relative to the published
                         defaults (the audit's p9 shortcut: "75% of the range if the default is the minimum, else 25%",
                         plus the quartiles and the centre) on calibrated generated worlds (`compose`): the median
                         share of the oracle's improvement each rule reaches must stay below `SHORTCUT_MAX`
                         (prespecified here, before any measurement of this module; the audit found 0.82)

`run_controls(workdir)` executes every control and returns {controls: {name: {passed, ...}}, passed, defined}.
Everything is measured on the real system (rapid worlds, the fresh-process evaluator, the generator).
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Sequence

import numpy as np

from . import fresh as FR
from . import learners as LR
from .distribution import Registry, build_registry
from .episodes import run_episode, score_episode
from .sampler import build_env

CONTROLS = ("process_stash", "file_stash", "config_smuggle", "memory_in_params", "pickle_checkpoint", "arch_edit",
            "calibration_shortcut")
SHORTCUT_MAX = 0.5          # a fixed public rule reaching half the oracle's improvement in the median world is a shortcut
GAIN_MIN = 0.02             # an attack "works" when it gains at least this much (normalised score) where undefended
TRAIN_SEEDS = tuple(760_000 + i for i in range(8))
EVAL_SEEDS = tuple(1_000_900 + i for i in range(6))


def _families(reg: Registry, k: int = 3) -> list[str]:
    """The `k` training families with the largest independently established weights gains (`instrument.establish`,
    establishment seeds): the attacks are trained and scored on all of them (one family's establishment seeds do
    not predict every seed of it)."""
    from .instrument import establish
    est = establish(reg)
    return sorted(est, key=lambda f: -est[f]["weights_gain"])[:k]


def _train(L, reg: Registry, families, rng, repeats: int = 3) -> None:
    for s in TRAIN_SEEDS:
        for f in ([families] if isinstance(families, str) else families):
            for _ in range(repeats):
                run_episode(build_env(reg[f], s, {}, {"mode": "assigned"}), L, learn=True, rng=rng, verify=False)


def _mean_score(reg: Registry, L, worlds, rng) -> float:
    was = L.frozen
    L.frozen = True
    v = float(np.mean([score_episode(reg, f, s, L, learn=False, rng=rng)["score"] for f, s in worlds]))
    L.frozen = was
    return v


def _fresh_mean(res: dict) -> float | None:
    if not res.get("ok"):
        return None
    return float(np.mean([x["score"] for x in res["scores"]]))


def _stash_control(arch: str, reg: Registry, families: list[str], workdir: str) -> dict:
    """Train `arch`; compare its untrained score, its in-process score and the fresh weights-only score (jailed and,
    for the file stash, unjailed) on unfamiliar worlds of the families."""
    seed = 7_001 if arch == "stateful" else 7_002
    rng = np.random.default_rng(seed)
    worlds = [(f, s) for f in families for s in EVAL_SEEDS]
    spec = [{"entry": f, "seed": s} for f, s in worlds]
    L = LR.make(arch, seed)
    if arch == "adv_file_stash":
        try:
            os.unlink(L._stash())
        except OSError:
            pass
    base = _mean_score(reg, L, worlds, rng)
    _train(L, reg, families, rng)
    inproc = _mean_score(reg, L, worlds, rng)
    man = FR.save_checkpoint(L, os.path.join(workdir, arch))
    jailed = FR.weights_only_eval(man, spec)
    out = {"untrained": base, "in_process": inproc, "in_process_gain": inproc - base,
           "fresh_jailed_ok": bool(jailed.get("ok")), "fresh_checks": jailed.get("checks"),
           "refused": jailed.get("refused"), "error": jailed.get("error")}
    fj = _fresh_mean(jailed)
    out["fresh_jailed_gain"] = None if fj is None else fj - base
    if arch == "adv_file_stash":
        open_ = FR.weights_only_eval(man, spec, jail=False)
        fu = float(np.mean([x["score"] for x in open_["scores"]])) if open_.get("scores") else None
        out["fresh_unjailed_gain"] = None if fu is None else fu - base
        out["attack_works_without_defence"] = fu is not None and fu - base >= GAIN_MIN
        try:
            os.unlink(L._stash())
        except OSError:
            pass
    worked = out["in_process_gain"] >= GAIN_MIN
    credited = out["fresh_jailed_gain"] is not None and abs(out["fresh_jailed_gain"]) > 1e-9
    out["passed"] = bool(worked and out["fresh_jailed_ok"] and not credited)
    return out


def _smuggle_control(reg: Registry, families: list[str], workdir: str) -> dict:
    rng = np.random.default_rng(7_003)
    L = LR.make("adv_config_smuggle", 7_003)
    seen = [(f, s) for f in families for s in TRAIN_SEEDS[:2]]
    base = _mean_score(reg, L, seen, rng)
    _train(L, reg, families, rng, repeats=2)
    inproc = _mean_score(reg, L, seen, rng)
    man = FR.save_checkpoint(L, os.path.join(workdir, "smuggle"))
    arch = json.load(open(os.path.join(man["dir"], "arch.json")))
    r = FR.weights_only_eval(man, [{"entry": f, "seed": s} for f, s in seen])
    return {"in_process_gain_on_seen_worlds": inproc - base, "table_in_configuration": bool(arch["cfg"].get("table")),
            "refused": r.get("refused"), "authorised_configuration": (r.get("checks") or {}).get(
                "authorised_configuration"),
            "passed": bool(inproc - base >= GAIN_MIN and arch["cfg"].get("table") and not r.get("ok") and
                           (r.get("checks") or {}).get("authorised_configuration") is False)}


def _tamper_controls(reg: Registry, families: list[str], workdir: str) -> dict:
    rng = np.random.default_rng(7_004)
    L = LR.make("weights", 7_004)
    _train(L, reg, families[0], rng, repeats=1)
    man = FR.save_checkpoint(L, os.path.join(workdir, "honest"))
    spec = [{"entry": families[0], "seed": EVAL_SEEDS[0]}]
    honest = FR.weights_only_eval(man, spec)
    out = {}
    for name, kind in (("memory_in_params", "memory_in_params"), ("pickle_checkpoint", "pickle"),
                       ("arch_edit", "arch_edit")):
        r = FR.weights_only_eval(FR.tampered_copy(man, kind, os.path.join(workdir, kind)), spec)
        out[name] = {"refused": r.get("refused"), "honest_checkpoint_accepted": bool(honest.get("ok")),
                     "passed": bool(honest.get("ok") and not r.get("ok") and r.get("refused"))}
    return out


def fixed_rules(u0: np.ndarray) -> dict[str, np.ndarray]:
    """The generator-shortcut rules, from the published defaults only."""
    u0 = np.asarray(u0, dtype=float)
    return {"anti_default": np.where(u0 < 0.5, 0.75, 0.25), "quartile_low": np.full_like(u0, 0.25),
            "quartile_high": np.full_like(u0, 0.75), "centre": np.full_like(u0, 0.5)}


class _Rule(LR.LeverLearner):
    arch = "fixed_rule"

    def __init__(self, rule: str):
        super().__init__(0)
        self.rule = rule

    def begin(self, ctx, rng):
        super().begin(ctx, rng)
        self._setting = fixed_rules(ctx["u0"])[self.rule]

    def act(self, k):
        return self._setting


def calibration_shortcut(n_structures: int = 24, seeds: Sequence[int] = (0, 1), *, structure_seed: int = 15_008) -> dict:
    """Median normalised score (share of the evaluator-side oracle's improvement over the defaults) of each fixed
    public rule on calibrated generated worlds; passes when every median is below `SHORTCUT_MAX`."""
    from . import curriculum as CU
    from .distribution import generated_entry
    from .episodes import normalised, references
    from .sampler import rapid_world
    rng = np.random.default_rng(structure_seed)
    rules = list(fixed_rules(np.zeros(1)))
    scores: dict[str, list[float]] = {r: [] for r in rules}
    worlds = 0
    tried = 0
    while worlds < n_structures * len(seeds) and tried < 6 * n_structures:
        tried += 1
        e = generated_entry(CU.random_structure(rng))
        for sd in seeds:
            if not rapid_world(e, sd, {}, None)._config.get("calibrated"):
                continue
            refs = references(e, sd)
            if refs["upper"] - refs["wait"] <= 1e-9:
                continue
            worlds += 1
            for r in rules:
                rec = run_episode(build_env(e, sd, {}, {"mode": "assigned"}), _Rule(r), learn=False,
                                  rng=np.random.default_rng(0))
                scores[r].append(normalised(rec["value"], refs))
    med = {r: float(np.median(v)) if v else None for r, v in scores.items()}
    ok = worlds > 0 and all(m is not None and m < SHORTCUT_MAX for m in med.values())
    return {"worlds": worlds, "median_share_of_oracle": med, "threshold": SHORTCUT_MAX,
            "share_above_0_75": {r: float(np.mean(np.array(v) > 0.75)) if v else None for r, v in scores.items()},
            "passed": bool(ok)}


def run_controls(workdir: str, *, registry: Registry | None = None, n_structures: int = 24) -> dict:
    """Every adversarial control of `CONTROLS` against the real instrument (module doc)."""
    reg = registry or build_registry()
    os.makedirs(workdir, exist_ok=True)
    fam = _families(reg)
    c = {"process_stash": _stash_control("stateful", reg, fam, workdir),
         "file_stash": _stash_control("adv_file_stash", reg, fam, workdir),
         "config_smuggle": _smuggle_control(reg, fam, workdir)}
    c.update(_tamper_controls(reg, fam, workdir))
    c["calibration_shortcut"] = calibration_shortcut(n_structures)
    LR.stash_clear()
    shutil.rmtree(LR.STASH_DIR, ignore_errors=True)
    return {"families": fam, "controls": c, "passed": sum(1 for v in c.values() if v.get("passed")),
            "defined": len(CONTROLS)}
