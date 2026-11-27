"""Executor factories for the G-env probes (loaded by ``python -m cam_live.harness.service --executor``)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAM = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(CAM, "src"))


def http_api():
    """The real generic HTTP API executor of workstream ES behind the ``api`` modality (no retries on its own)."""
    from cam_live.harness.failures import RetryPolicy
    from cam_live.harness.http_api import ApiModality, HttpApiExecutor
    return ApiModality([HttpApiExecutor(retry=RetryPolicy(rules={}))])
