import pytest
from harness_helpers import make_sandbox


@pytest.fixture()
def sandbox(tmp_path, scratch_bb):
    s = make_sandbox(tmp_path, scratch_bb)
    yield s
    s.srv.stop()
    s.lab.close()
