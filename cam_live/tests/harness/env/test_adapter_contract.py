"""Environment-adapter contract kit: generic conformance checks for ANY ``contracts.EnvironmentAdapter``.

Runs against the built-in computer environment and against the diagnostic LabWorld adapter. To run it against a
different harness, point ``CAM_LIVE_ADAPTER_FACTORY`` at ``module:factory`` where ``factory(run_dir)`` returns
``(adapter, policy, task)`` — see docs/HARNESS_ADAPTER_GUIDE.md::

    CAM_LIVE_ADAPTER_FACTORY=my_pkg.my_env:make .venv/bin/python -m pytest \
        cam_live/tests/harness/env/test_adapter_contract.py -q --rootdir=cam_live
"""
import importlib
import json
import os

import pytest

from cam_live.contracts import (EXEC_INTERFACE_VERSION, ActionResult, ActionSpec, AuthorizationDecision, AuthVerdict,
                                EnvironmentAdapter, FailureClass, Level, Modality, Observation, OutcomeObservation,
                                OutcomeQuery, OutcomeStatus, Quantity, ResultStatus, TaskSpec, ToolDescriptor, Trust,
                                new_id)

from eb_support import GrantPolicy


def _computer(run_dir):
    from cam_live.harness.env import ComputerEnvironment
    from cam_live.harness.internal import InternalExecutor
    policy = GrantPolicy()
    env = ComputerEnvironment(run_dir, policy)
    env.register(InternalExecutor(run_dir, clock=env.clock))
    return env, policy, TaskSpec("obj_contract", "contract check", mode="staging")


def _labworld(run_dir):
    from cam_live.diagnostics.world import AllowAllPolicy, LabWorld, lab_task
    world = LabWorld()
    return world, AllowAllPolicy(), lab_task(world)


def _external(run_dir):
    mod, _, attr = os.environ["CAM_LIVE_ADAPTER_FACTORY"].partition(":")
    return getattr(importlib.import_module(mod), attr)(run_dir)


FACTORIES = {"computer_environment": _computer, "labworld": _labworld}
if os.environ.get("CAM_LIVE_ADAPTER_FACTORY"):
    FACTORIES = {"external": _external}


@pytest.fixture(params=sorted(FACTORIES))
def rig(request, tmp_path):
    adapter, policy, task = FACTORIES[request.param](str(tmp_path))
    yield adapter, policy, task
    close = getattr(adapter, "close", None)
    if callable(close):
        close()


def _minimal_args(schema):
    """Smallest argument object a tool's schema accepts (required properties only)."""
    out = {}
    for name in schema.get("required", []):
        p = schema.get("properties", {}).get(name, {})
        t = p.get("type")
        if "enum" in p:
            out[name] = p["enum"][0]
        elif t in ("integer", "number"):
            out[name] = p.get("minimum", 0)
        elif t == "boolean":
            out[name] = False
        elif t == "array":
            out[name] = []
        elif t == "object":
            out[name] = {}
        else:
            out[name] = "x"
    return out


def test_adapter_satisfies_the_environment_contract(rig):
    adapter, policy, task = rig
    assert isinstance(adapter, EnvironmentAdapter)
    assert adapter.interface_version.split(".")[0] == EXEC_INTERFACE_VERSION.split(".")[0]
    tools = adapter.discover_tools()
    assert tools and all(isinstance(t, ToolDescriptor) for t in tools)
    assert len({t.name for t in tools}) == len(tools)
    for t in tools:
        assert t.modality in Modality.ALL and t.level in Level.ALL and t.description.strip()
        assert t.side_effects in ("none", "local", "external_read", "external_write")
        assert t.arg_schema.get("type", "object") == "object" and (t.available or t.unavailable_reason)
    obs = adapter.observe()
    assert isinstance(obs, list) and all(isinstance(o, Observation) for o in obs)
    assert all(o.provenance.trust == Trust.UNTRUSTED and o.obs_id and o.kind for o in obs)
    events, cur = adapter.poll_events("")
    assert isinstance(events, list) and isinstance(cur, str) and all(isinstance(e, Observation) for e in events)
    again, cur2 = adapter.poll_events(cur)
    assert again == [] and cur2 == cur                                # polling at the head returns nothing new
    json.dumps(adapter.get_resource_state(), default=str)
    assert all(isinstance(q, OutcomeQuery) for q in adapter.get_pending_outcomes())
    assert all(isinstance(q, Quantity) for q in adapter.get_observable_metrics())
    caps = adapter.get_capabilities()
    assert caps["interface_version"] == adapter.interface_version and "live" in caps and "provenance" in caps
    assert adapter.measure([]) == []
    not_due = OutcomeQuery(new_id("q"), "contract.unknown.metric", horizon_s=10 ** 9, issued_at=10 ** 10)
    (m,) = adapter.measure([not_due])
    assert isinstance(m, OutcomeObservation) and m.status == OutcomeStatus.PENDING and m.value is None


def test_adapter_validates_and_refuses_without_authorization(rig):
    adapter, policy, task = rig
    bogus = ActionSpec(new_id("act"), "no.such.operation", Modality.API)
    assert adapter.validate_action(bogus)
    tool = next(t for t in adapter.discover_tools() if t.available and t.modality in (Modality.INTERNAL, Modality.API)
                and t.name != "program")
    action = ActionSpec(new_id("act"), tool.name, tool.modality, level=tool.level, arguments=_minimal_args(tool.arg_schema))
    assert adapter.validate_action(action) == []
    for auth in (None, AuthorizationDecision(AuthVerdict.DENY, action.content_hash(), ["contract check"])):
        res = adapter.execute(action, auth)
        assert isinstance(res, ActionResult) and res.status == ResultStatus.DENIED
        assert res.failure_class in FailureClass.HARD and res.action_id == action.action_id
    need = adapter.execute(action, AuthorizationDecision(AuthVerdict.NEEDS_APPROVAL, action.content_hash(), approval_id="a1"))
    assert need.status == ResultStatus.NEEDS_APPROVAL
    res = adapter.execute(action, policy.authorize(action, task, {}))
    assert isinstance(res, ActionResult) and res.action_id == action.action_id and res.status in (
        ResultStatus.OK, ResultStatus.FAILED, ResultStatus.TIMEOUT)
    assert res.ended >= res.started and all(isinstance(o, Observation) for o in res.observations)
    assert all(o.provenance.trust == Trust.UNTRUSTED for o in res.observations)
    bad = ActionSpec(new_id("act"), tool.name, tool.modality, arguments={"__not_an_argument__": object.__name__ * 3,
                                                                         **{k: None for k in tool.arg_schema.get("required", [])}})
    if tool.arg_schema.get("required"):
        assert adapter.validate_action(bad)


def test_adapter_returns_no_reward(rig):
    adapter, policy, task = rig

    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield str(k).lower()
                yield from keys(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                yield from keys(v)

    tool = next(t for t in adapter.discover_tools() if t.available and t.modality in (Modality.INTERNAL, Modality.API)
                and t.name != "program")
    action = ActionSpec(new_id("act"), tool.name, tool.modality, level=tool.level, arguments=_minimal_args(tool.arg_schema))
    res = adapter.execute(action, policy.authorize(action, task, {}))
    seen = set(keys(res.to_dict())) | set(keys(adapter.get_resource_state())) | set(keys(
        [o.to_dict() for o in adapter.observe()])) | set(keys([q.to_dict() for q in adapter.get_observable_metrics()]))
    assert not {k for k in seen if "reward" in k or k in ("bonus", "score")}
    assert adapter.get_capabilities().get("features", {}).get("reward", False) is False
