"""Fixtures for the EC tests (sandbox, background jobs, visual executor, grounding)."""
import os
import shutil
import threading
import uuid

import pytest

from cam_live import contracts as c
from cam_live.harness import sandbox as sbx

_OPT_IN = ("browser", "privileged", "network")


def pytest_collection_modifyitems(config, items):
    """browser / privileged / network tests run only when named with -m (gpu / slow: see the root conftest)."""
    expr = config.getoption("-m") or ""
    for item in items:
        for mark in _OPT_IN:
            if mark in item.keywords and mark not in expr:
                item.add_marker(pytest.mark.skip(reason=f"run with -m {mark}"))
        if "privileged" in item.keywords and os.geteuid() != 0:
            item.add_marker(pytest.mark.skip(reason="needs root"))


class Policy:
    """A recording stand-in for PolicyEngine.check_primitive (the authorization service belongs to EA)."""

    def __init__(self, rule=None):
        self.rule = rule or (lambda op, action: c.AuthVerdict.ALLOW)
        self.calls = []
        self._lock = threading.Lock()

    def __call__(self, op, action):
        verdict = self.rule(op, action)
        with self._lock:
            self.calls.append((op.op, dict(op.arguments), verdict))
        return c.AuthorizationDecision(verdict=verdict, reasons=[] if verdict == c.AuthVerdict.ALLOW else ["test policy"])

    def ops(self, name=None):
        return [x for x in self.calls if name is None or x[0] == name]


@pytest.fixture(scope="session")
def sbx_base():
    base = f"/var/tmp/cam-live-sbx-test-{uuid.uuid4().hex[:10]}"
    os.makedirs(base, exist_ok=True)
    os.chmod(base, 0o711)
    yield base
    shutil.rmtree(base, ignore_errors=True)


@pytest.fixture
def sbx_cfg(sbx_base):
    def make(**kw):
        kw.setdefault("allow_same_uid", os.geteuid() != 0)      # non-root hosts: test the weaker mode, reported
        return sbx.SandboxConfig(base_dir=sbx_base, **kw)
    return make


@pytest.fixture
def sandbox(sbx_base, sbx_cfg):
    made = []

    def make(**kw):
        s = sbx.Sandbox(os.path.join(sbx_base, "ws", uuid.uuid4().hex[:10]), sbx_cfg(**kw))
        if not s.available:
            pytest.skip(f"sandbox unavailable on this host: {s.unavailable_reason}")
        made.append(s)
        return s
    yield make
    for s in made:
        s.close()


@pytest.fixture
def make_ctx(tmp_path):
    def make(policy=None, run_dir=None, clock=None, **kw):
        sink = []
        ctx = c.ExecContext(clock=clock or c.WallClock(), check=policy if policy is not None else Policy(),
                            obs_sink=sink.append, run_dir=str(run_dir or tmp_path / "run"),
                            cancel=kw.pop("cancel", threading.Event()), **kw)
        os.makedirs(ctx.run_dir, exist_ok=True)
        ctx.sink = sink
        return ctx
    return make


def action(op, modality=c.Modality.CODE, timeout_s=60.0, verify=None, **arguments):
    return c.ActionSpec(action_id=c.new_id("act"), operation=op, modality=modality, arguments=arguments,
                        timeout_s=timeout_s, verify=verify or c.VerificationCondition())


@pytest.fixture
def act():
    return action


@pytest.fixture
def policy_cls():
    return Policy


def session_processes(sid_or_uid_filter=None):
    """Live (non-zombie) processes whose command line contains ``sid_or_uid_filter``."""
    out = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode("utf-8", "replace")
            with open(f"/proc/{name}/stat", "rb") as f:
                state = f.read().rsplit(b")", 1)[1].split()[0]
        except OSError:
            continue
        if state != b"Z" and sid_or_uid_filter in cmd:
            out.append((int(name), cmd))
    return out


@pytest.fixture
def procs_matching():
    return session_processes
