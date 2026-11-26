"""The six levels of generalization (MASTER_SPEC Part 15.1; U.1, U.2, SP-15.1).

    split = build_level(level, registry)      # train items, test items, and the checks that make it that level
    res = run_level(split, learner_arch, seeds)   # train on train items only, score test items in the evaluator

| Level | unfamiliar element                       | how the split is built from the registry                       |
|-------|------------------------------------------|----------------------------------------------------------------|
| 1     | new random seeds                         | same training families; test seeds from the validation range   |
| 2     | new parameter distributions              | same families and seeds range; training at difficulty levels 0  |
|       |                                          | and 1 only, testing at level 2 (shifted noise, distractors,    |
|       |                                          | opponents, resources)                                           |
| 3     | unseen combinations of familiar          | test families whose measured primitive set is covered by the   |
|       | mechanisms                               | training families' primitives but whose primitive combination  |
|       |                                          | (motif set and motif-edge set) no training family has          |
| 4     | entirely unfamiliar mechanisms           | test families with a primitive or an engine that no training   |
|       |                                          | family has (measured fingerprints)                              |
| 5     | new domains, action interfaces and       | test entries of another tier (a Tier 2 domain configuration or |
|       | independently implemented simulators     | a held-out Tier 3 profile) plus 4.0 mechanism tasks (scm, abm, |
|       |                                          | techsys: other simulators, the rig.* tool interface), trained  |
|       |                                          | on Tier 1 rapid families only; needs a Tier 2 or Tier 3 entry  |
| 6     | authorised real-world objectives         | `Level6Adapter`: authorisation-gated, disabled by default       |

Levels 1-4 use the lever-interface learners of `learners` (scores normalised by the evaluator-side references);
Level 5 is run with the architecture-neutral `crucible.experience.ReferenceLearner`, because no lever interface
exists in a domain or civilisation world, and is scored by each world's own verifier. A level is *executable* only
if its split exists in this checkout, passes its checks (disjointness at the level's element, training items only
in training) and the protocol ran end to end with at least `hstats.MIN_SEEDS` learner seeds.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from . import hstats as HS
from . import learners as LR
from .distribution import Registry, build_registry, entry_fingerprint, thaw
from .episodes import lever_entries, run_episode, score_episode
from .sampler import build_env

LEVELS = (1, 2, 3, 4, 5, 6)


@dataclass
class Split:
    level: int
    train: list[dict]
    test: list[dict]
    checks: dict = field(default_factory=dict)
    why: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.train) and bool(self.test) and all(self.checks.values())

    def to_json(self) -> dict:
        return {"level": self.level, "train": self.train, "test": self.test, "checks": self.checks, "why": self.why,
                "ok": self.ok}


def _fp_sets(reg: Registry, eid: str) -> tuple[set, set, set]:
    fp = entry_fingerprint(reg[eid])
    motifs = set(map(str, fp.get("motifs") or []))
    edges = set(map(str, fp.get("motif_edges") or []))
    eng = set(map(str, fp.get("engine") or [])) if isinstance(fp.get("engine"), (list, tuple, set)) else \
        {str(fp.get("engine"))}
    return motifs, edges, eng


def build_level(level: int, reg: Registry | None = None, *, n_train: int = 12, seeds_per: int = 2) -> Split:
    reg = reg or build_registry()
    fams = lever_entries(reg, "train")
    rng = np.random.default_rng(1500 + level)
    if level in (1, 2):
        tr = sorted(fams[i] for i in rng.permutation(len(fams))[:n_train])
        train = [{"entry": f, "seed": 400 + k, "level": "0" if (level == 1 or k % 2 == 0) else "1"}
                 for k, f in enumerate(tr * seeds_per)]
        if level == 1:
            test = [{"entry": f, "seed": 1_000_300 + k} for k, f in enumerate(tr[:8])]
            checks = {"same_families": {t["entry"] for t in test} <= {t["entry"] for t in train},
                      "new_seeds": not ({(t["entry"], t["seed"]) for t in test} &
                                        {(t["entry"], t["seed"]) for t in train}),
                      "test_in_validation": all(reg.partition_of(t["entry"], t["seed"]) == "validation"
                                                for t in test)}
            return Split(1, train, test, checks)
        test = [{"entry": f, "seed": 1_000_400 + k, "level": "2"} for k, f in enumerate(tr[:8])]
        from .distribution import difficulty_for
        shifted = [t for t in test if difficulty_for(reg[t["entry"]], "2")]
        checks = {"same_families": {t["entry"] for t in test} <= {t["entry"] for t in train},
                  "test_level_unseen": not ({t.get("level") for t in train} & {"2"}),
                  "parameters_shifted": len(shifted) == len(test),
                  "test_in_validation": all(reg.partition_of(t["entry"], t["seed"]) == "validation" for t in test)}
        return Split(2, train, [t for t in shifted], checks)
    if level in (3, 4):
        sets = {f: _fp_sets(reg, f) for f in fams}
        order = [fams[i] for i in rng.permutation(len(fams))]
        tr, held = order[:n_train], order[n_train:]
        tm = set().union(*(sets[f][0] for f in tr))
        te_ = set().union(*(sets[f][2] for f in tr))
        combos = {(frozenset(sets[f][0]), frozenset(sets[f][1])) for f in tr}
        if level == 3:
            pick = [f for f in held if sets[f][0] <= tm and (frozenset(sets[f][0]), frozenset(sets[f][1])) not in combos
                    and sets[f][1] - set().union(*(sets[g][1] for g in tr))]
            checks = {"primitives_familiar": all(sets[f][0] <= tm for f in pick),
                      "combination_unseen": all((frozenset(sets[f][0]), frozenset(sets[f][1])) not in combos
                                                for f in pick),
                      "families_disjoint": not (set(pick) & set(tr))}
        else:
            pick = [f for f in held if (sets[f][0] - tm) or (sets[f][2] - te_)]
            checks = {"mechanism_unfamiliar": all((sets[f][0] - tm) or (sets[f][2] - te_) for f in pick),
                      "families_disjoint": not (set(pick) & set(tr))}
        train = [{"entry": f, "seed": 500 + k} for k, f in enumerate(tr * seeds_per)]
        test = [{"entry": f, "seed": 1_000_500 + k} for k, f in enumerate(pick[:8])]
        checks["test_in_validation"] = all(reg.partition_of(t["entry"], t["seed"]) == "validation" for t in test)
        return Split(level, train, test, checks, why="" if pick else "no held-out family qualifies")
    if level == 5:
        rapid_train = [w["id"] for w in reg.worlds(tier=1, family_partition="train", source="crucible.rapid")]
        new_domain = [w["id"] for w in reg.worlds(tier=2) + reg.worlds(tier=3) if w["family_partition"] in
                      ("evaluation",)]
        mech = [w["id"] for w in reg.worlds(tier=1, source="crucible.mechanisms") if w["family_partition"] == "train"]
        train = [{"entry": f, "seed": 700 + k} for k, f in enumerate(sorted(rapid_train)[:n_train])]
        test = [{"entry": e, "seed": 1_000_700 + k} for k, e in enumerate(sorted(new_domain)[:4])]
        test += [{"entry": e, "seed": 1_000_800 + k} for k, e in enumerate(sorted(mech)[:3])]
        train_tiers = {reg[t["entry"]]["tier"] for t in train}
        checks = {"new_domain_present": any(reg[t["entry"]]["tier"] not in train_tiers for t in test),
                  "new_action_interface": bool({a for t in test for a in reg[t["entry"]]["action_kinds"]} -
                                               {a for t in train for a in reg[t["entry"]]["action_kinds"]}),
                  "independent_simulators": bool({reg[t["entry"]]["engine"] for t in test} -
                                                 {reg[t["entry"]]["engine"] for t in train})}
        return Split(5, train, test, checks, why="" if checks["new_domain_present"] else
                     "no Tier 2 domain configuration or Tier 3 evaluation profile is available in this checkout")
    raise ValueError(f"levels 1-5 have simulation splits; got {level}")


def _lever_run(split: Split, arch: str, seeds: Sequence[int], reg: Registry) -> dict:
    from .distribution import difficulty_for
    rows_u, rows_t = [], []
    audit = reg.audit(split.train)
    for s in seeds:
        L = LR.make(arch, s)
        rng = np.random.default_rng(9000 + s)
        before = []
        for t in split.test:
            d = difficulty_for(reg[t["entry"]], t.get("level", "0"))
            before.append(score_episode(reg, t["entry"], t["seed"], L, learn=False, rng=rng, difficulty=d)["score"])
        for t in split.train:
            d = difficulty_for(reg[t["entry"]], t.get("level", "0"))
            for _ in range(3):
                run_episode(build_env(reg[t["entry"]], t["seed"], d, {"mode": "assigned"}), L, learn=True, rng=rng)
        L.frozen = True
        after = []
        for t in split.test:
            d = difficulty_for(reg[t["entry"]], t.get("level", "0"))
            after.append(score_episode(reg, t["entry"], t["seed"], L, learn=False, rng=rng, difficulty=d)["score"])
        rows_u.append(before)
        rows_t.append(after)
    res = HS.crossed_diff(np.array(rows_u), np.array(rows_t), paired_seeds=True)
    return {"learner": arch, "seeds": list(seeds), "training_audit": audit, "transfer_gain": res,
            "untrained_mean": float(np.mean(rows_u)), "trained_mean": float(np.mean(rows_t))}


def _reference_run(split: Split, seeds: Sequence[int], reg: Registry, steps: int = 48) -> dict:
    """Level 5: the architecture-neutral reference learner, trained on Tier 1 and scored by each test world's
    own verifier (no lever interface is assumed)."""
    from ...experience.loop import run_learner
    from ...experience.reference import ReferenceLearner
    from .episodes import verified_value
    rows = []
    audit = reg.audit(split.train)
    for s in seeds:
        L = ReferenceLearner(seed=s)
        for t in split.train[:4]:
            env = build_env(reg[t["entry"]], t["seed"], {}, {"mode": "assigned"})
            run_learner(env, L, steps)
        L.freeze(True)
        vals = []
        for t in split.test:
            env = build_env(reg[t["entry"]], t["seed"], {}, {"mode": "assigned"})
            run_learner(env, L, steps)
            try:
                vals.append(verified_value(env.verify()))
            except ValueError:
                vals.append(float("nan"))
        rows.append(vals)
    return {"learner": "crucible.experience.ReferenceLearner", "seeds": list(seeds), "training_audit": audit,
            "verified_values": rows, "scored_worlds": int(np.isfinite(np.array(rows)).all(axis=0).sum())}


def run_level(split: Split, seeds: Sequence[int] = (0, 1, 2), arch: str = "weights",
              reg: Registry | None = None) -> dict:
    reg = reg or build_registry()
    out = {"level": split.level, "split": split.to_json()}
    if not split.ok:
        out.update(executed=False, why=split.why or f"split checks failed: {split.checks}")
        return out
    try:
        if split.level == 5:
            r = _reference_run(split, seeds, reg)
            executed = r["scored_worlds"] == len(split.test) and not r["training_audit"]["violations"]
        else:
            r = _lever_run(split, arch, seeds, reg)
            executed = not r["training_audit"]["violations"] and len(seeds) >= HS.MIN_SEEDS
    except Exception as e:  # noqa: BLE001 - a level that cannot run is reported with its classified cause
        from .integrity import classify_failure
        out.update(executed=False, why=f"{type(e).__name__}: {str(e)[:300]}", failure=classify_failure(e))
        return out
    out.update(result=r, executed=bool(executed))
    return out


# ====================================================================== Level 6
class Level6Disabled(PermissionError):
    """Level 6 is off: it must be enabled explicitly and authorised by an operator document."""


METHODOLOGY = "docs/v5/LEVEL6_METHODOLOGY.md"
MIN_EXPERIMENTS_FOR_CLAIM = 3


class Level6Adapter:
    """The Level 6 evaluation adapter over `crucible.realworld.RealWorldBridge` (docs/v5/LEVEL6_METHODOLOGY.md).

    Disabled by default: it does nothing unless constructed with ``enable=True`` *and* ``CRUCIBLE_LEVEL6=1`` is set
    in the environment *and* an operator authorisation document is given (`crucible.realworld.load_authorization`);
    the bridge then enforces the document's scope, budgets, approvals and logging. It must run in a process that
    has not hosted synthetic worlds (the bridge refuses otherwise). An objective is an authorised experiment; the
    learner acts through `act(request)` (HTTP requests the bridge checks) and the outcome is computed only by the
    experimenter's measurement function (`RealWorldBridge.measure`), never from what the learner reports. No
    generalisation claim is produced unless at least `MIN_EXPERIMENTS_FOR_CLAIM` measured, logged experiments exist
    (`claim()`)."""

    def __init__(self, authorization_path: str | None = None, *, log_path: str | None = None, enable: bool = False,
                 approve=None):
        self.enabled = False
        self.bridge = None
        self.experiments: list[dict] = []
        if not enable or os.environ.get("CRUCIBLE_LEVEL6") != "1":
            raise Level6Disabled("Level 6 is disabled by default: pass enable=True and set CRUCIBLE_LEVEL6=1")
        if not authorization_path:
            raise Level6Disabled("Level 6 requires an operator authorisation document")
        from ...realworld.bridge import RealWorldBridge
        self.bridge = RealWorldBridge.from_file(authorization_path, log_path=log_path, approve=approve)
        self.enabled = bool(self.bridge.enabled)

    def run(self, policy, measure, *, name: str, steps: int = 5) -> dict:
        """Run `policy(reader_or_bridge, step) -> {"method", "url", "body"?} | None` for `steps` steps through the
        bridge, then measure the outcome with `measure(reader)` (the experimenter's function)."""
        if not self.enabled:
            raise Level6Disabled("authorisation is not in force")
        acts = 0
        for k in range(int(steps)):
            req = policy(self.bridge.reader(), k)
            if req is None:
                continue
            self.bridge.request(str(req["method"]), str(req["url"]), body=req.get("body"))
            acts += 1
        res = self.bridge.measure(name, measure)
        rec = {"name": name, "actions": acts, "value": res.value, "log_head": res.log_head,
               "authorization_sha256": res.authorization_sha256}
        self.experiments.append(rec)
        return rec

    def claim(self) -> dict:
        n = len(self.experiments)
        if n < MIN_EXPERIMENTS_FOR_CLAIM:
            return {"claim": None, "why": f"{n} executed authorised experiments < {MIN_EXPERIMENTS_FOR_CLAIM}"}
        return {"claim": "measured outcomes of authorised experiments", "experiments": n,
                "scope": "the authorised staging systems of the document only"}

    def close(self):
        if self.bridge is not None:
            self.bridge.close()
