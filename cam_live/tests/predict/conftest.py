"""Local fixtures for tests/predict (Workstream B)."""
import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(1)          # tiny models: one thread is as fast and immune to a busy machine
    yield


@pytest.fixture
def gen():
    return torch.Generator().manual_seed(1234)


_MEASURED = {}


@pytest.fixture
def measure():
    """Record a measured number. With CAM_LIVE_B_MEASURE=<path> the collected numbers are written there as JSON
    at the end of the session (used to produce docs/evidence/B/measured.json)."""
    def _rec(name, value):
        _MEASURED[name] = value
    return _rec


def pytest_sessionfinish(session, exitstatus):
    import json
    import os
    path = os.environ.get("CAM_LIVE_B_MEASURE")
    if path and _MEASURED:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(dict(sorted(_MEASURED.items())), f, indent=1, sort_keys=True)


# ---- stage 2: the real (debug-scale) model in LabWorld -------------------------------------------------------------
class LabRig:
    """A debug-scale CamLiveModel with B's heads, a store of honestly collected LabWorld lifetimes, and helpers to
    run the runtime prediction path in a fresh world. Nothing here stands in for code under test."""

    def __init__(self, tmp_path, n_lifetimes=4, n_decisions=12, seed=0, mechanism_seed=None, **section):
        import dataclasses
        import random

        from cam_live.contracts import OutcomeKind, OutcomeQuery, RunFiles
        from cam_live.diagnostics.memstore import MemoryStore
        from cam_live.diagnostics.world import LabVerifier, LabWorld, lab_task
        from cam_live.predict import labdata as L
        from cam_live.predict.predictor import Predictor, at_horizons
        from cam_live.predict.registry import ExperimentRegistry, PredictionRegistry

        torch.manual_seed(seed)
        self.L = L
        self.model, self.heads = L.build(seed, **section)
        self.store = MemoryStore()
        self.lifetimes = L.collect(self.store, range(n_lifetimes), n_decisions, mechanism_seed=mechanism_seed) \
            if n_lifetimes else []
        self.records = [r for lt in self.lifetimes for r in lt.records]
        self.world = LabWorld(L.lab_config(seed=1000 + seed))
        self.clock = self.world.clock
        self.task = lab_task(self.world)
        self.verifier = LabVerifier(self.world)
        self.deadline = self.world.t0 + self.world.cfg.horizon_s
        self.registry = PredictionRegistry(str(tmp_path / RunFiles.PREDICTIONS), self.clock)
        self.experiments = ExperimentRegistry(str(tmp_path / RunFiles.EXPERIMENTS), self.registry, self.clock, seed=3)
        self.fast = self.model.new_fast_state()
        self.predictor = Predictor(self.model, self.fast, self.registry, self.clock, heads=self.heads)
        self.rng = random.Random(seed)
        self._tmpl = at_horizons(OutcomeQuery("", "lab.adopters.from.{action}", OutcomeKind.COUNT,
                                              description="adopters attributed to this intervention within the "
                                                          "attribution window", goal_relevant=True),
                                 [L.SHORT_HORIZON_S, 2 * L.DAY])
        self._replace = dataclasses.replace

    def queries(self, action):
        """Outcome questions the agent asks about a candidate before acting (two horizons for interventions)."""
        if action.operation != "lab.act":
            return []
        return [self._replace(t, exposure=float(action.arguments["effort"])) for t in self._tmpl]

    def context(self):
        from cam_live.diagnostics.collect import obs_item
        return [obs_item(o) for o in self.world.observe()]

    def candidates(self):
        from cam_live.predict.candidates import schema_candidates
        return schema_candidates(self.world.discover_tools(), per_tool=6, id_prefix=f"a{self.rng.randrange(10**9)}")

    def evaluate(self, context=None, candidates=None, **kw):
        kw.setdefault("objective_id", self.task.objective_id)
        kw.setdefault("deadline", self.deadline)
        kw.setdefault("objective_scale", float(self.world.cfg.target))
        kw.setdefault("validate", self.world.validate_action)
        return self.predictor.evaluate_candidates(context if context is not None else self.context(),
                                                  self.task.objective,
                                                  candidates if candidates is not None else self.candidates(),
                                                  self.queries, **kw)

    def execute(self, action):
        from cam_live.diagnostics.world import AllowAllPolicy
        return self.world.execute(action, AllowAllPolicy().authorize(action, self.task, {}))

    def check_signature(self, outcome):
        from cam_live.diagnostics.world import LabVerifier
        return any(LabVerifier(lt.world).check_signature(outcome) for lt in self.lifetimes) or \
            self.verifier.check_signature(outcome)


@pytest.fixture
def lab_rig(tmp_path):
    def make(**kw):
        return LabRig(tmp_path, **kw)
    return make
