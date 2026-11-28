"""DX: change request INT3A-5 (G2L-8) — the record-level held-out split cannot be empty once an objective has enough
records, and a promotion without held-out evidence says so.

Before: every record was held out by an independent hash (val_fraction 0.15), so an objective of N records had no
held-out record with probability 0.85^N (0.10 for N = 14) and its promotion was refused for "no independent validation
metric supplied". Now the k-th stored record of an objective is held out iff floor(u + (k+1) f) > floor(u + k f)
(u = seeded hash of the objective): exactly every 1/f-th record."""
import math

import pytest
from conftest import build_record

from cam_live.config import CamLiveConfig
from cam_live.credit import checkpoint as C
from cam_live.credit.learner import VAL_PREFIX, LearnerSettings
from cam_live.learning_api import TIMESCALE_FREQUENT

F = 0.15


class Always:
    timescale = TIMESCALE_FREQUENT
    needs_latents = False

    def __init__(self, name="always"):
        self.name, self.weight = name, 1.0

    def eligible(self, record):
        return True

    def loss(self, model, batch, store):
        return None, {}


def _lab(lab_factory, **settings):
    cfg = CamLiveConfig.debug()
    st = dict(val_split="record", val_fraction=F)
    st.update(settings)
    return lab_factory(collect=False, cfg=cfg, world=False, policy=False, credit=False,
                       settings=LearnerSettings.from_cfg(cfg, **st))


def _fill(store, n_objectives, sizes, prefix=""):
    i = 0
    with store.batch():
        for j in range(n_objectives):
            for _ in range(sizes(j)):
                store.append(build_record(i, objective=f"{prefix}obj{j}", prefix=prefix))
                i += 1


def test_every_objective_with_enough_records_has_a_held_out_record_and_the_fraction_is_kept(lab_factory):
    lab = _lab(lab_factory)
    L, st = lab.learner, lab.store
    _fill(st, 60, lambda j: 7 + (j % 12))                      # objectives of 7 .. 18 records
    by_obj = {}
    for r in st.all():
        by_obj.setdefault(r.objective_id, []).append(L.is_validation(r))
    none = [o for o, v in by_obj.items() if not any(v)]
    assert not none, f"{len(none)} objectives without a held-out record"
    for o, v in by_obj.items():                                 # exactly floor(u + N f) - floor(u) of each objective
        assert math.floor(len(v) * F) <= sum(v) <= math.ceil(len(v) * F)
    total = sum(sum(v) for v in by_obj.values()) / sum(len(v) for v in by_obj.values())
    assert abs(total - F) < 0.02


def test_the_split_is_deterministic_stable_under_appends_and_one_sided(lab_factory):
    lab = _lab(lab_factory)
    L, st = lab.learner, lab.store
    p = L.register(Always())
    _fill(st, 10, lambda j: 9)
    first = {r.exp_id: L.is_validation(r) for r in st.all()}
    L.refresh_eligibility(full=True)
    _fill(st, 10, lambda j: 6, prefix="later_")                 # later records of other objectives ...
    with st.batch():                                            # ... and of the same objectives
        for j in range(10):
            st.append(build_record(1000 + j, objective=f"obj{j}", prefix="z"))
    L2 = _lab(lab_factory).learner                              # a fresh learner, same seed, different store object
    L2.store = st
    again = {e: L2.is_validation(st.get(e)) for e in first}
    assert again == first                                       # stored records keep their side
    L.refresh_eligibility(full=True)
    for r in st.all():
        tr, va = r.eligible.get(p.name), r.eligible.get(VAL_PREFIX + p.name)
        assert (tr, va) in ((True, False), (False, True)) and va == L.is_validation(r)
    other_seed = _lab(lab_factory, seed=1).learner
    other_seed.store = st
    assert {e: other_seed.is_validation(st.get(e)) for e in first} != first


def test_small_objectives_and_the_objective_split_are_unchanged_in_kind(lab_factory):
    lab = _lab(lab_factory)
    L, st = lab.learner, lab.store
    _fill(st, 40, lambda j: 1 + (j % 6))                       # 1 .. 6 records: a held-out record is possible, not sure
    held = [sum(L.is_validation(r) for r in st.trajectory(f"obj{j}")) for j in range(40)]
    assert all(h <= 1 for h in held) and 0 < sum(held) < 40
    lo = _lab(lab_factory, val_split="objective").learner       # whole objectives by hash, as before
    lo.store = st
    for j in range(40):
        sides = {lo.is_validation(r) for r in st.trajectory(f"obj{j}")}
        assert len(sides) == 1


def test_a_promotion_without_held_out_evidence_says_so_explicitly(lab_factory, tmp_path):
    cfg = CamLiveConfig.debug()
    for frac, expect in ((0.0, False), (0.4, True)):
        lab = lab_factory(n_worlds=4, n_decisions=5, cfg=cfg, policy=False, credit=False,
                          settings=LearnerSettings.from_cfg(cfg, val_split="record", val_fraction=frac))
        L = lab.learner
        L.refresh_eligibility(full=True)
        mgr = C.CheckpointManager(str(tmp_path / f"ck{frac}"))
        dec, _, _ = C.promote_learner(mgr, L, retention={"lab/base/neg_nll": -1.0})
        assert dec.held_out["evidence"] is expect
        assert set(dec.held_out["validation_records"]) == {p.name for p in L.providers}
        if expect:
            assert dec.promoted and dec.held_out["providers_with_loss"]
            assert sum(dec.held_out["validation_records"].values()) > 0
        else:
            assert not dec.promoted
            assert any(r.startswith("no held-out evidence") for r in dec.reasons), dec.reasons
            assert sum(dec.held_out["validation_records"].values()) == 0
        log = mgr.promotion_log()[-1]
        assert log["held_out"]["evidence"] is expect
    # caller-supplied validation metrics: nothing is said about held-out records
    dec = mgr.promote(mgr.latest().checkpoint_id, validation={"x": 1.0}, retention={"y": 1.0})
    assert dec.held_out == {}
