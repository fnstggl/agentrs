"""DX regression tests for the slow-step guard (CL-03; G2_learn.md G2L-4, G2L-5) and the learner's eligibility flags.

Probe losses with an exactly known effect of one adapter step on their held-out value (the construction of
test_gcl_fixes.py): training minimises X = 10 + s * mean(adapters); evaluation reports eval_sign * X."""
import hashlib
import json

import pytest
import torch

from cam_live.config import CamLiveConfig
from cam_live.credit.learner import VAL_PREFIX, LearnerSettings
from cam_live.learning_api import TIMESCALE_FREQUENT, TIMESCALE_SLOW, UpdatePlan
from cam_live.nn.api import GROUP_ADAPTERS


class AdapterProbe:
    timescale = TIMESCALE_SLOW
    needs_latents = False

    def __init__(self, name, s=100.0, eval_sign=-1.0, spread=0.0, eligible=None, eval_none=False):
        self.name, self.s, self.eval_sign, self.spread, self.weight = name, s, eval_sign, spread, 1.0
        self._eligible = eligible
        self.eval_none = eval_none            # after a training step the held-out loss cannot be computed
        self.trained = False

    def eligible(self, record):
        return True if self._eligible is None else bool(self._eligible(record))

    def loss(self, model, batch, store):
        ps = model.param_groups()[GROUP_ADAPTERS]
        n = sum(p.numel() for p in ps)
        x = 10.0 + self.s * sum(p.sum() for p in ps) / n
        if self.spread:
            f = sum(int(hashlib.sha1(r.exp_id.encode()).hexdigest()[:4], 16) / 65536.0 for r in batch.records)
            x = 10.0 + (x - 10.0) * (1.0 + self.spread * (f / max(len(batch.records), 1) - 0.5))
        if model.training:
            self.trained = True
            return x, {}
        if self.eval_none and self.trained:
            return None, {}
        return self.eval_sign * x, {}


class HonoursOwnFlag:
    """A frequent-timescale provider that, like ``predict.providers._Base._permitted``, treats
    ``record.eligible[self.name] == False`` as an exclusion, and whose own eligibility can change later."""
    timescale = TIMESCALE_FREQUENT
    needs_latents = False

    def __init__(self, name="flagged"):
        self.name, self.weight, self.open = name, 1.0, set()

    def eligible(self, record):
        return record.eligible.get(self.name, True) and record.objective_id in self.open

    def loss(self, model, batch, store):
        return None, {}


def _lab(lab_factory, **settings):
    cfg = CamLiveConfig.debug()
    cfg.optim.lr_adapters = 1e-4
    st = dict(slow_guard=True, val_fraction=0.4, val_batch=32, slow_batch_size=4)
    st.update(settings)
    torch.manual_seed(0)
    return lab_factory(n_worlds=8, n_decisions=6, cfg=cfg, world=False, policy=False, credit=False,
                       settings=LearnerSettings.from_cfg(cfg, **st))


def _adapters(m):
    return [p.detach().clone() for p in m.param_groups()[GROUP_ADAPTERS]]


def _same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


def _slow(L):
    rec = L.learn(UpdatePlan(trigger="test", timescales=(TIMESCALE_SLOW,)))[-1]
    return rec, json.loads(rec.notes or "{}")


def test_without_held_out_records_a_slow_step_is_deferred_not_taken(lab_factory):
    lab = _lab(lab_factory, val_fraction=0.0)
    L, m = lab.learner, lab.model
    L.register(AdapterProbe("probe", eval_sign=-1.0))          # a step that would raise held-out loss
    L.refresh_eligibility(full=True)
    assert not L._validation_cells()
    w0 = _adapters(m)
    rec, notes = _slow(L)
    assert _same(w0, m.param_groups()[GROUP_ADAPTERS])
    assert rec.param_groups == [] and notes["deferred"] is True and notes["guard"] is None
    assert "held-out" in notes["skipped"]
    assert L.counters["slow_deferred"] == 1 and L.counters["slow"] == 0
    assert L.slow_guard_state["deferred"] == 1
    # with held-out evidence the same learner takes guarded steps again
    L.settings.val_fraction = 0.4
    L.refresh_eligibility(full=True)
    L.provider("probe").eval_sign = +1.0
    rec, notes = _slow(L)
    assert notes["guard"]["accepted"] and rec.param_groups == [GROUP_ADAPTERS]


def test_one_providers_degradation_is_not_hidden_by_anothers_gain(lab_factory):
    lab = _lab(lab_factory)
    L, m = lab.learner, lab.model
    L.register(AdapterProbe("improves", s=100.0, eval_sign=+1.0, spread=0.5))
    L.register(AdapterProbe("degrades", s=50.0, eval_sign=-1.0, spread=0.5))
    L.refresh_eligibility(full=True)
    w0 = _adapters(m)
    rec, notes = _slow(L)
    g = notes["guard"]
    assert not g["accepted"] and _same(w0, m.param_groups()[GROUP_ADAPTERS])      # behaviour first
    by = g["by_provider"]
    assert set(g["judged"]) == {"improves", "degrades"} and g["rule"] == "per_provider"
    assert by["improves"]["paired_change"] < 0 < by["degrades"]["paired_change"]
    assert g["mean_over_cells"]["change"] < 0                  # the average over providers improved ...
    assert by["degrades"]["paired_change"] > by["degrades"]["tolerance"]
    assert not by["degrades"]["accepted"] and by["improves"]["accepted"]
    assert not g["accepted"] and g["reverted"] and g["binding_provider"] == "degrades"   # ... the step is refused
    assert _same(w0, m.param_groups()[GROUP_ADAPTERS]) and rec.param_groups == []
    # per-provider drift is tracked separately once steps are accepted
    L.provider("degrades").eval_sign = +1.0
    rec, notes = _slow(L)
    assert notes["guard"]["accepted"]
    dp = L.slow_guard_state["drift_by_provider"]
    assert set(dp) == {"improves", "degrades"} and all(v < 0 for v in dp.values())
    assert L.slow_guard_state["drift"] == pytest.approx(max(dp.values()))


def test_a_loss_without_held_out_records_of_its_own_does_not_move_the_adapters(lab_factory):
    lab = _lab(lab_factory)
    L, m = lab.learner, lab.model
    judge = L.register(AdapterProbe("judge", s=0.0, eval_sign=+1.0))      # held out, but no adapter gradient
    L.register(AdapterProbe("unjudged", s=100.0, eval_sign=-1.0, eligible=lambda r: not L.is_validation(r)))
    L.refresh_eligibility(full=True)
    cells = L._validation_cells()
    assert {k[0] for k in cells} == {"judge"}
    w0 = _adapters(m)
    rec, notes = _slow(L)
    assert _same(w0, m.param_groups()[GROUP_ADAPTERS]) and rec.param_groups == []
    assert notes["excluded_without_held_out"] == ["unjudged"] and notes["deferred"] is True
    # once the judged loss itself reaches the adapters it alone drives the step, and the verdict names the other
    judge.s = 100.0
    rec, notes = _slow(L)
    g = notes["guard"]
    assert notes["excluded_without_held_out"] == ["unjudged"] and g["accepted"]
    assert g["judged"] == ["judge"] and g["unjudged"] == ["unjudged"]
    assert rec.param_groups == [GROUP_ADAPTERS]


def test_a_held_out_loss_that_cannot_be_computed_after_the_step_rejects_it(lab_factory):
    lab = _lab(lab_factory)
    L, m = lab.learner, lab.model
    L.register(AdapterProbe("vanishes", eval_sign=+1.0, eval_none=True))
    L.refresh_eligibility(full=True)
    w0 = _adapters(m)
    rec, notes = _slow(L)
    assert _same(w0, m.param_groups()[GROUP_ADAPTERS]) and rec.param_groups == []
    g = notes["guard"]
    assert not g["accepted"] and g["reverted"] and any("not computable" in r for r in g["reasons"])


def test_held_out_flags_survive_a_refresh_and_late_eligibility_is_picked_up(lab_factory):
    """Providers that read their own ``eligible`` flag (as predict.providers do) used to see the learner's held-out
    exclusion as permanent: validation records lost ``val:<name>`` at the next refresh, and a record judged
    ineligible once never became trainable."""
    lab = _lab(lab_factory, val_split="record")
    L, st = lab.learner, lab.store
    p = L.register(HonoursOwnFlag())
    objs = sorted({r.objective_id for r in st.all()})
    p.open = set(objs[:4])
    L.refresh_eligibility(full=True)
    n_train, n_val = st.count(loss=p.name), st.count(loss=VAL_PREFIX + p.name)
    assert n_train > 0 and n_val > 0
    L.refresh_eligibility(full=True)                        # a second refresh changes nothing
    assert (st.count(loss=p.name), st.count(loss=VAL_PREFIX + p.name)) == (n_train, n_val)
    p.open = set(objs)                                      # more records become eligible later
    L.refresh_eligibility(full=True)
    n_all = sum(1 for r in st.all() if r.objective_id in p.open)
    assert st.count(loss=p.name) + st.count(loss=VAL_PREFIX + p.name) == n_all
    assert st.count(loss=p.name) > n_train and st.count(loss=VAL_PREFIX + p.name) >= n_val
    # what the provider sees when its held-out loss is evaluated: its own eligibility, not the training exclusion
    val = st.sample(4, loss=VAL_PREFIX + p.name, seed=0)
    assert val and all(r.eligible[p.name] is False for r in val)
    assert all(p.eligible(r) for r in L._as_validation(p, val))
    assert all(r.eligible[p.name] is False for r in st.get_many([r.exp_id for r in val]))   # store untouched
