"""Shared test setup for CAM Live. MASTER-OWNED (add per-directory conftest.py files for local fixtures)."""
import contextlib
import fcntl
import os
import sys
import time

import pytest

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# INT2-5 / INT3B: the whole suite is collected in ONE process with ``--import-mode=importlib`` (pyproject addopts):
# test modules get unique names from their path, so two ``test_providers.py`` in different directories no longer
# clash. That mode does not put a test's directory on sys.path, so the directories whose sibling helper modules
# tests import (``rt_helpers``, ``int_helpers``, ``ea_helpers``, ...) are appended here, once, in a fixed order.
_TESTS = os.path.dirname(os.path.abspath(__file__))
for _d in ("credit", "runtime", "integration", "nn", os.path.join("harness", "env"), os.path.join("harness", "auth"),
           os.path.join("harness", "skills")):
    _p = os.path.join(_TESTS, _d)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)
# ``from conftest import build_record`` (tests/credit) must always mean tests/credit/conftest.py, whichever conftest
# files pytest has loaded before: pytest imports a conftest whose directory is on sys.path as the top-level module
# ``conftest`` (harness/env, harness/auth, harness/skills), which replaced the credit one in a whole-suite run (INT2-5
# clash 2). The credit module is set as ``conftest`` now (module-level imports at collection) and again before every
# test of tests/credit (imports inside tests).
_CREDIT_CONFTEST = None
if os.path.exists(os.path.join(_TESTS, "credit", "conftest.py")):
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("conftest", os.path.join(_TESTS, "credit", "conftest.py"))
    _CREDIT_CONFTEST = _ilu.module_from_spec(_spec)
    sys.modules["conftest"] = _CREDIT_CONFTEST
    _spec.loader.exec_module(_CREDIT_CONFTEST)


def _credit_conftest_for(path) -> None:
    if _CREDIT_CONFTEST is not None and os.path.join(_TESTS, "credit") + os.sep in str(path):
        sys.modules["conftest"] = _CREDIT_CONFTEST


def pytest_collect_file(file_path, parent):
    _credit_conftest_for(file_path)                 # module-level ``from conftest import ...`` at collection
    return None


def pytest_runtest_setup(item):
    _credit_conftest_for(item.path)                 # ``from conftest import ...`` inside a credit test
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")


def pytest_collection_modifyitems(config, items):
    """Opt-in tests run only when asked for with -m (keeps the default suite fast, CPU-only and offline).
    ``gpu`` / ``slow``: every such mark on a test must be named. ``browser`` / ``privileged`` / ``network``: a test
    carrying several of them runs when any one is named (``-m privileged`` runs privileged browser tests)."""
    expr = config.getoption("-m") or ""
    for item in items:
        for mark in ("gpu", "slow"):
            if mark in item.keywords and mark not in expr:
                item.add_marker(pytest.mark.skip(reason=f"run with -m {mark}"))
        env_marks = [m for m in ("browser", "privileged", "network") if m in item.keywords]
        if env_marks and not any(m in expr for m in env_marks):
            item.add_marker(pytest.mark.skip(reason=f"run with -m {env_marks[0]}"))


@contextlib.contextmanager
def gpu_lock(timeout_s: float = 1800.0, min_free_gb: float = 0.0):
    """Serialise heavy GPU work with every other tenant of this GPU. Hold for <= ~15 minutes. Never wrap pytest
    itself in ``flock`` on this file: tests take the lock here and would deadlock."""
    from cam_live.config import GPU_LOCK_PATH
    os.makedirs(os.path.dirname(GPU_LOCK_PATH), exist_ok=True)
    f = open(GPU_LOCK_PATH, "a+")
    t0 = time.time()
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.time() - t0 > timeout_s:
                f.close()
                pytest.skip("shared GPU lock not acquired in time")
            time.sleep(5)
    try:
        if min_free_gb:
            import torch
            free = torch.cuda.mem_get_info()[0] / 2**30
            if free < min_free_gb:
                pytest.skip(f"only {free:.1f} GB GPU memory free, need {min_free_gb}")
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


@pytest.fixture
def gpu_heavy():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    with gpu_lock(min_free_gb=14.0):
        yield
        torch.cuda.empty_cache()
