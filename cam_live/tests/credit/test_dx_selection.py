"""DX: evidence-based estimator selection runs where the integrated agent learns (RC-08; G2_learn.md G2L-6).

A small *test agent* does what ``CamLiveAgent`` does with experiments: it registers randomised experiments in its own
hash-chained ``predict.registry.ExperimentRegistry`` (never in the experience store), runs them as randomised
decisions inside LabWorld lifetimes (decision ``experiment_id`` + ``diagnostics['arm']``, logged propensities),
links outcomes through the ``OutcomeLinker``, lets the verifier close each objective, and hands the matured
experience to the ``ContinualLearner`` with a ``CreditAssigner``. The only wiring it adds is the change request
docs/change_requests/DX.md asks of agent.py: ``assigner.attach_experiments(registry)``.

Checked end to end through the real store, linker, learner, RUDDER / COCOA providers and selector:
* the registry's experiments reach the experience store before the learner's held-out split is refreshed, so the
  experiment lifetimes are held out of training (``holdout_experiments=True``) and the selection runs on them;
* with the agent's current setting (``holdout_experiments=False``: every lifetime is trained on) and with
  one-unit experiments (the agent's pattern today) the selection does not run, and the selection log says why;
* nothing in-sample is ever scored.
"""
import json
import random

import pytest

from cam_live.config import CamLiveConfig
from cam_live.contracts import Distribution, OutcomeKind, OutcomeQuery, PredictionRecord, new_id
from cam_live.credit.labkit import (Arms, build_lab, collect_lifetime, door_candidates, door_config,
                                    door_weight, mk_action, run_door_training, weighted_policy)
from cam_live.credit.learner import LearnerSettings
from cam_live.credit.select import EstimatorSelector
from cam_live.diagnostics.world import LabWorld
from cam_live.learning_api import UpdatePlan
from cam_live.predict.registry import ExperimentRegistry, PredictionRegistry

BETA = 2e-5


class TestAgent:
    """The experiment bookkeeping of the integrated agent, on D's LabWorld lab (debug-scale real model)."""
    __test__ = False

    def __init__(self, tmp_path, *, holdout_experiments=True, attach=True, seed=0, val_split="objective"):
        cfg = CamLiveConfig.debug()
        self.dir = tmp_path
        self.selector_log = str(tmp_path / "credit_selection.jsonl")
        self.lab = build_lab(str(tmp_path / "store.sqlite"), cfg, world=False, policy=False, credit=True, beta=BETA,
                             selector=None, settings=LearnerSettings.from_cfg(
                                 cfg, batch_size=16, val_fraction=0.15, val_split=val_split,
                                 holdout_experiments=holdout_experiments))
        self.lab.assigner.selector = EstimatorSelector(log_path=self.selector_log, clock=self.lab.clock)
        self.predictions = PredictionRegistry(str(tmp_path / "prediction_registry.jsonl"), self.lab.clock)
        self.experiments = ExperimentRegistry(str(tmp_path / "experiment_registry.jsonl"), self.predictions,
                                              self.lab.clock, seed=seed)
        if attach:                                   # the one line the change request adds to agent.py
            self.lab.assigner.attach_experiments(self.experiments)
        self.seed = seed

    def register_experiment(self, e):
        now = self.lab.clock.now()
        q = OutcomeQuery(query_id=new_id("q"), metric="lab.return", kind=OutcomeKind.REAL, horizon_s=86400.0,
                         issued_at=now, goal_relevant=True)
        pred = PredictionRecord(prediction_id=new_id("pred"), query=q, made_at=now,
                                dist=Distribution("normal", {"loc": 0.0, "scale": 1.0}, mean=0.0, std=1.0))
        rec = self.experiments.register(question="does preparing the last channel first change the later return?",
                                        hypothesis="the arms differ in matured return", predictions=[pred],
                                        arms=["unlock", "wait"], assignment="randomized",
                                        experiment_id=f"door-{self.seed}-{e}")
        return rec.experiment_id

    def run_experiment(self, e, n_units, n_decisions=4):
        xid = self.register_experiment(e)
        arms = Arms(xid, {"unlock": lambda: mk_action("lab.unlock"),
                          "wait": lambda: mk_action("wait", "internal", seconds=3600.0)}, {"unlock": 0.5, "wait": 0.5})
        out = []
        for i in range(n_units):
            s = 100_000 * (self.seed + 1) + 1000 * e + i
            world = LabWorld(door_config(seed=1000 * (self.seed + 1) + e), self.lab.clock)
            world.rng = random.Random(s)
            out.append(collect_lifetime(self.lab.store, self.lab.linker, world, weighted_policy(door_weight, seed=s),
                                        n_decisions, seed=s, candidates_fn=door_candidates, arms=arms,
                                        tag=f"{self.seed}_{e}_{i}"))
        return xid, out

    def learn(self, rounds=3):
        st = self.lab.store
        ids = [r.exp_id for r in st.all()]
        for i in range(rounds):
            self.lab.learner.learn(UpdatePlan(trigger="replay", max_steps=4, budget_s=1e9,
                                              exp_ids=ids if i == 0 else []))

    def log(self):
        try:
            with open(self.selector_log) as fh:
                return [json.loads(x) for x in fh if x.strip()]
        except FileNotFoundError:
            return []

    def close(self):
        self.lab.close()


@pytest.fixture
def agents(tmp_path):
    made = []

    def make(**kw):
        sub = tmp_path / f"agent{len(made)}"
        sub.mkdir()
        made.append(TestAgent(sub, **kw))
        return made[-1]
    yield make
    for a in made:
        a.close()


def test_the_agents_experiments_reach_the_store_and_the_selection_runs_on_held_out_lifetimes(agents):
    ag = agents()
    run_door_training(ag.lab, 2, 6, 4, seed=0)               # estimator training experience (not experiments)
    runs = dict(ag.run_experiment(e, 6) for e in range(2))
    st, L = ag.lab.store, ag.lab.learner
    assert st.experiments() == []                             # recorded only in the agent's registry so far
    ag.learn()
    # the registry's experiments are in the store, unchanged
    stored = {e.experiment_id: e for e in st.experiments()}
    assert set(stored) == set(runs)
    for xid in runs:
        assert stored[xid].to_dict() == ag.experiments.get(xid).to_dict()
    # ... early enough for the held-out split: no experiment lifetime was trained on by anything
    x_objs = {r.objective_id for xid in runs for r in st.experiment_records(xid)}
    assert x_objs and not (x_objs & L.trained_objectives)
    assert not (x_objs & ag.lab.rudder.trained_objectives) and ag.lab.rudder.trained_objectives
    # the selection ran on held-out randomised evidence, logged with the per-experiment report
    ev = ag.lab.assigner.last_evaluation
    assert ev.n_compared == 2 and ev.n_evidence == 2, ev.reason
    rows = [r for r in ag.log() if r.get("selection")]
    assert len(rows) == 1 and len(ag.log()) == 1, ag.log()     # one selection; later passes: same measurement
    sel, report = rows[-1]["selection"], rows[-1]["evaluation"]
    assert sel["n_randomized_experiments"] == 2 and not sel["excluded_not_held_out"]
    assert all(h["held_out"] is True and h["n_excluded_trained"] == 0 for h in sel["holdout"].values())
    assert {x["experiment_id"] for x in report["experiments"] if x["compared"]} == set(runs)
    for x in report["experiments"]:
        for pr in x["comparisons"]:
            n = sum(u["held_out"] for u in pr["units"].values())
            assert pr["compared"] and pr["held_out"] is True and n == len(runs[x["experiment_id"]])
            assert "rudder" in pr["estimates"] and pr["paired_n"]["rudder"] == n
    assert sel["default_after"] == ag.lab.assigner.default == ev.default_after
    assert L.credit_selection["n_evidence"] == 2 and L.status()["credit_selection"] == L.credit_selection
    # a second pass with unchanged evidence neither re-runs the selection nor writes another row
    n_rows = len(ag.log())
    ag.learn(rounds=1)
    assert len(ag.log()) == n_rows and not ag.lab.assigner.last_evaluation.ran


def test_with_every_lifetime_trained_on_the_selection_does_not_run_and_says_why(agents):
    ag = agents(holdout_experiments=False, val_split="record")    # the integrated agent's current settings
    run_door_training(ag.lab, 1, 4, 4, seed=1)
    runs = dict(ag.run_experiment(e, 6) for e in range(2))
    ag.learn(rounds=6)
    st, L, A = ag.lab.store, ag.lab.learner, ag.lab.assigner
    assert {e.experiment_id for e in st.experiments()} == set(runs)
    x_objs = {r.objective_id for xid in runs for r in st.experiment_records(xid)}
    assert x_objs <= A.trained_objectives()                    # all of them were trained on
    ev = A.last_evaluation
    assert not ev.ran and ev.n_evidence == 0 and A.default == "simple"
    assert "held-out units" in ev.reason
    rows = ag.log()
    assert rows and rows[-1]["selection"] is None and rows[-1]["evaluation"]["ran"] is False
    for x in rows[-1]["evaluation"]["experiments"]:
        pr = x["comparisons"][0]
        assert not pr["compared"] and "lifetime trained on" in pr["reason"]
        assert sum(u["lifetime_trained_on"] for u in pr["units"].values()) == len(runs[x["experiment_id"]])
    assert not A.selector.evidence                             # nothing in-sample was ever added


def test_one_unit_experiments_are_reported_as_insufficient(agents):
    ag = agents()
    xids = [ag.run_experiment(e, 1)[0] for e in range(3)]       # one randomised decision per experiment
    ag.learn(rounds=1)
    ev = ag.lab.assigner.last_evaluation
    assert ev.n_randomized == 3 and ev.n_compared == 0
    for x in ev.experiments:
        assert x["experiment_id"] in xids and x["decisions"] == 1 and not x["compared"]
        assert "(< 2" in x["reason"]
    # CR: their held-out units reach the pooled rule, which promotes nothing on 3 units and says so
    assert ev.n_units == 3 and ev.default_after == "simple"
    sel = ag.lab.assigner.selector.history[-1]
    assert sel.mode == "pooled_units" and not sel.changed
    assert any("only 3 held-out randomised units" in r for r in sel.reasons)


def test_without_an_attached_source_nothing_changes(agents):
    ag = agents(attach=False)
    ag.run_experiment(0, 4)
    ag.learn(rounds=1)
    assert ag.lab.store.experiments() == [] and ag.log() == []
    assert ag.lab.assigner.last_evaluation is None and ag.lab.learner.credit_selection == {}
    # the API also works when called directly with the records
    ev = ag.lab.assigner.evaluate_experiments(ag.experiments.records())
    assert ev.n_recorded == 1 and ev.n_experiments == 1
    assert ev.default_after == "simple" and ev.n_units <= 4         # CR: too few pooled units to promote anything


@pytest.mark.slow
def test_integrated_agent_with_the_change_request_line_records_and_evaluates_its_experiments(tmp_path):
    """The real ``CamLiveAgent`` (tiny CPU configuration) driven by the real runtime against LabWorld, with the one
    line of docs/change_requests/DX.md applied here at construction: its experiments reach the experience store and
    every learning update evaluates them; what the selection could not do is in the selection log."""
    import torch
    from int_helpers import Recorder, agent_config, run_lifetime

    from cam_live.agent import CamLiveAgent
    from cam_live.contracts import SimClock

    torch.set_num_threads(2)
    clock = SimClock()
    agent = CamLiveAgent(agent_config(policy_update_every=6), str(tmp_path / "agent"), clock=clock)
    try:
        agent.assigner.attach_experiments(agent.experiments)          # the change request (agent.py, INT3A)
        rec = Recorder(agent)
        lab = {"outcome_horizon_s": 1800.0, "mean_delay_s": (300.0, 900.0)}
        for s in (11, 12):
            run_lifetime(rec, clock, str(tmp_path), s, max_steps=40, max_decisions=22, lab_kw=lab)
        reg = {e.experiment_id for e in agent.experiments.records()}
        on_decisions = {r.decision.experiment_id for r in agent.store.all() if r.decision.experiment_id}
        assert reg and on_decisions <= reg
        assert {e.experiment_id for e in agent.store.experiments()} == reg
        ev = agent.assigner.last_evaluation
        assert ev is not None and ev.n_experiments == len(reg)
        rows = [json.loads(x) for x in open(agent.selector.log_path) if x.strip()]
        assert rows and all(("evaluation" in r) for r in rows)
        for r in rows:                                                # never scored in-sample
            for h in ((r.get("selection") or {}).get("holdout") or {}).values():
                assert h["held_out"] is True
        if not ev.ran:
            assert ev.reason and all(x.get("compared") or x.get("reason") for x in ev.experiments)
        # the learner's status keeps the last pass (a later poll() rewrites trigger_state; it did in one run)
        assert agent.learner.status()["credit_selection"].get("default") == agent.assigner.default
    finally:
        agent.close()
