"""Local fixtures for Workstream D tests: builders for real contract records (no doubles of D's own code)."""
import os
import sys

import pytest

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from cam_live.contracts import (ActionResult, ActionSpec, CandidateEvaluation, Decision, ExperienceRecord, Level,  # noqa: E402
                                OutcomeKind, OutcomeQuery, ResultStatus)


def build_record(i: int, *, ts: float = None, objective: str = "obj1", family: str = "famA", provenance: str = "live",
                 n_cand: int = 3, chosen: int = 0, probs=None, status: str = ResultStatus.OK, query: bool = True,
                 horizon: float = 100.0, eligible=("world",), level: str = Level.INTERVENTION, kind: str = OutcomeKind.COUNT,
                 experiment_id: str = "", arm: str = "", duration: float = 5.0, artifacts=None, model_version: str = "m1",
                 text_pad: int = 0, prefix: str = "") -> ExperienceRecord:
    ts = 1000.0 + 10.0 * i if ts is None else ts
    cands = [ActionSpec(action_id=f"{prefix}a{i}_{j}", operation="op.do", level=level,
                        arguments={"choice": j, "arm": arm} if arm else {"choice": j}, rationale="x" * text_pad)
             for j in range(n_cand)]
    probs = list(probs) if probs is not None else [1.0 / n_cand] * n_cand
    dec = Decision(decision_id=f"{prefix}d{i}", ts=ts, objective_id=objective, chosen=cands[chosen],
                   candidates=[CandidateEvaluation(action=c) for c in cands], probabilities=probs, chosen_index=chosen,
                   exploration="randomized" if experiment_id else "thompson", model_version=model_version,
                   reward_version="r1", experiment_id=experiment_id, level=level,
                   diagnostics={"arm": arm} if arm else {})
    res = ActionResult(action_id=cands[chosen].action_id, status=status, started=ts, ended=ts + duration,
                       artifacts=dict(artifacts or {}))
    rec = ExperienceRecord(exp_id=f"{prefix}e{i}", objective_id=objective, decision=dec, result=res,
                           context_item_refs=[], goal_text="reach the target", duration_s=duration,
                           eligible={k: True for k in eligible}, objective_family=family, provenance=provenance)
    if query:
        q = OutcomeQuery(query_id=f"{prefix}q{i}", metric=f"m.from.{cands[chosen].action_id}", kind=kind,
                         horizon_s=horizon, issued_at=ts + duration, decision_id=dec.decision_id, goal_relevant=True,
                         experiment_id=experiment_id)
        rec.pending_queries.append(q)
    return rec


@pytest.fixture
def make_record():
    return build_record


# ---- stage 2: the real model (debug scale) in LabWorld --------------------------------------------------------------
def make_lab(tmp_path, *, n_worlds=10, n_decisions=6, seed=3, family="lab/base", collect=True, lab_cfg=None, **kw):
    """A debug-scale CamLiveModel with heads, a durable store, the linker and the learner, plus (optionally) a few
    LabWorld lifetimes collected through the linker."""
    import torch
    from cam_live.credit.labkit import build_lab, collect_family
    from cam_live.diagnostics.world import LabConfig
    torch.manual_seed(0)
    lab = build_lab(str(tmp_path / "store.sqlite"), **kw)
    if collect:
        base = lab_cfg or LabConfig(n_channels=4, family=family)
        collect_family(lab.store, lab.linker, base, n_worlds, n_decisions, seed=seed, clock=lab.clock)
    return lab


@pytest.fixture
def lab_factory(tmp_path):
    labs = []

    def build(**kw):
        sub = tmp_path / f"lab{len(labs)}"
        sub.mkdir()
        labs.append(make_lab(sub, **kw))
        return labs[-1]
    yield build
    for lab in labs:
        lab.close()
