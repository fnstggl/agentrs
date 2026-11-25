"""Local sandboxed code execution and the end-to-end path: generated code action -> sandbox -> result tokens."""
from __future__ import annotations

import pytest
import torch

from sem_helpers import make_context
from superlearner.cam_unified.interfaces import Action, EntityRegistry, ObsKind, TokenKind
from superlearner.cam_unified.semantic.sandbox import (CodeSandbox, LocalCodeTools, SandboxLimits, SandboxUnavailable,
                                                       crucible_sandbox_capabilities)

pytestmark = pytest.mark.skipif(not crucible_sandbox_capabilities().get("available"),
                                reason="Crucible Landlock+seccomp sandbox unavailable")


@pytest.fixture(scope="module")
def sbx():
    return CodeSandbox(limits=SandboxLimits(timeout_s=3.0))


def test_script_stdout_returns_as_exec_result(sbx):
    r = sbx.run(code="import numpy as np\nprint(int(np.arange(7).sum()) * 2)\n", t=2.5, caused_by="a_1")
    assert r.ok and r.duration == 0.0 and r.meta["backend"] == "crucible"
    (o,) = r.observations
    assert o.kind == ObsKind.EXEC_RESULT and "42" in o.content and o.caused_by == "a_1" and o.t == 2.5


def test_exception_becomes_error_observation(sbx):
    r = sbx.run(code="x = {}\nprint('before')\nx['missing']\n")
    assert not r.ok and "KeyError" in r.error
    (o,) = r.observations
    assert o.kind == ObsKind.ERROR and "KeyError" in o.content and "before" in o.content


def test_function_entry_returns_json_result(sbx):
    files = {"solver.py": "def solve(inp):\n    return {'total': sum(inp['xs']), 'n': len(inp['xs'])}\n"}
    r = sbx.run(files=files, entry="solver.py:solve", args={"xs": [1, 2, 3, 4]})
    assert r.ok and r.observations[0].meta["result"] == {"total": 10, "n": 4}


def test_isolation_network_filesystem_and_timeout(sbx):
    net = sbx.run(code="import socket\nsocket.socket()\n")
    assert not net.ok and ("PermissionError" in net.error or "Operation not permitted" in net.observations[0].content)
    fs = sbx.run(code="print(open('/etc/passwd').read()[:10])\n")
    assert not fs.ok and "Permission" in fs.observations[0].content
    sub = sbx.run(code="import subprocess\nsubprocess.run(['ls'])\n")
    assert not sub.ok
    slow = sbx.run(code="while True:\n    pass\n")
    assert not slow.ok and "Timeout" in slow.error
    bad = sbx.run(files={"../escape.py": "print(1)", "main.py": "print(2)"})
    assert not bad.ok and "InvalidProgram" in bad.error
    empty = sbx.run(files={}, entry="x.py:f")                # an agent's empty program is a failed outcome
    assert not empty.ok and empty.observations[0].kind == ObsKind.ERROR


IDIOMS = {   # program -> (ok, text expected in the observation, error type)
    "main_guard": ("def main():\n    print(sum(range(11)))\n\nif __name__ == '__main__':\n    main()\n", True, "55", None),
    "exit0": ("print(55)\nimport sys\nsys.exit(0)\n", True, "55", None),
    "exit_none": ("print(55)\nraise SystemExit\n", True, "55", None),
    "exit3": ("print('partial')\nimport sys\nsys.exit(3)\n", False, "partial", "NonZeroExit"),
    "exit_msg": ("import sys\nsys.exit('fatal: bad input')\n", False, "fatal: bad input", "NonZeroExit"),
    "exception": ("x = {}\nprint('before')\nx['missing']\n", False, "before", "KeyError"),
    "no_newline": ("print(55, end='')", True, "55", None),
}


@pytest.mark.parametrize("backend", ["crucible", "subprocess"])
def test_scripts_run_as_main_with_exit_status_on_both_backends(backend):
    """main.py runs as __main__ (guarded programs execute), sys.exit(0) is success, other statuses are failures, and
    the two backends agree; the runner's internal result never leaks into the observation."""
    sb = CodeSandbox(backend=backend, allow_weak=True, limits=SandboxLimits(timeout_s=5.0))
    for name, (code, ok, text, err) in IDIOMS.items():
        r = sb.run(code=code)
        (o,) = r.observations
        assert r.ok is ok, (backend, name, o.content)
        assert text in o.content and "__cam_exit__" not in o.content and "imported" not in o.content, (name, o.content)
        assert o.kind == (ObsKind.EXEC_RESULT if ok else ObsKind.ERROR)
        if err:
            assert r.error.startswith(err), (name, r.error)
        if ok:
            assert o.meta["result"] is None and "result:" not in o.content
    sib = sb.run(files={"main.py": "import helper\nprint(helper.f(4))\n", "helper.py": "def f(x):\n    return x * 10\n"})
    assert sib.ok and "40" in sib.observations[0].content
    assert not sb.run(files={"main.py": "print(1)", "__cam_runner__.py": "x = 1"}).ok      # reserved name


def test_weak_backend_only_when_explicit():
    with pytest.raises(SandboxUnavailable):
        CodeSandbox(backend="subprocess")
    weak = CodeSandbox(backend="subprocess", allow_weak=True, limits=SandboxLimits(timeout_s=3.0))
    r = weak.run(code="print(sum(range(5)))")
    assert r.ok and "10" in r.observations[0].content and r.meta["backend"] == "subprocess"
    f = weak.run(files={"m.py": "def f(a):\n    return a * 3\n"}, entry="m.py:f", args=14)
    assert f.ok and f.observations[0].meta["result"] == 42
    assert not weak.run(code="while True: pass").ok


def test_generated_code_action_executes_and_result_returns_as_tokens(gen, tok, cfg, sbx):
    tools = LocalCodeTools(sbx)
    tool_lists = [tools.tools(), tools.tools()]
    from superlearner.cam_unified.testing import sample_observations
    from superlearner.cam_unified.interfaces import Observation
    task = Observation(ObsKind.TEXT, "Task: write a Python program that prints the sum of the integers 1..10.",
                       t=0.0, source="task")
    obs_lists = [[task] + sample_observations(2, seed=1), [task]]
    R, goal, belief = make_context(cfg, 2, seed=21)
    regs = [EntityRegistry(cfg.max_entities, seed=b) for b in range(2)]
    obs = tok.observations(obs_lists, regs, [0.0, 0.0])
    tt = tok.tools(tool_lists)
    lists = gen.tool_lists_with_builtins(tool_lists)
    mask = torch.zeros(2, len(lists[0]), dtype=torch.bool, device=gen.device)
    mask[0, 0] = True                                     # run_python
    mask[1, 1] = True                                     # run_function
    acts = gen.sample(R, goal, belief, tt, obs, tool_lists, obs_lists, tool_mask=mask,
                      generator=torch.Generator(device=gen.device).manual_seed(0))
    assert [a.tool for a in acts] == ["run_python", "run_function"]
    assert isinstance(acts[0].args["code"], str) and len(acts[0].args["code"]) >= 1
    assert acts[0].logp_parts["n_tokens"] > 0               # the program text was written by the decoder
    results = [tools.execute(a, t=0.0) for a in acts]
    for a, r in zip(acts, results):                          # real execution: success or a real failure
        (o,) = r.observations
        assert o.kind in (ObsKind.EXEC_RESULT, ObsKind.ERROR) and o.caused_by == a.action_id
    res_tokens = tok.observations([r.observations for r in results], regs, [0.0, 0.0])
    assert res_tokens.mask.sum(1).tolist() == [cfg.sem_tokens_per_obs] * 2
    assert (res_tokens.kind[res_tokens.mask] == TokenKind.OBS_SEM).all()
    # a correct program for the task (e.g. a demonstration) runs to the right answer through the same executor
    good = Action("run_python", {"code": "print(sum(range(1, 11)))"})
    lp, _ = gen.log_prob(R[:1], goal.select([0]), belief.select([0]), tt.select([0]), obs.select([0]),
                         tool_lists[:1], obs_lists[:1], [good])
    assert torch.isfinite(lp).all() and float(lp[0]) < 0
    r = tools.execute(good)
    assert r.ok and "55" in r.observations[0].content
