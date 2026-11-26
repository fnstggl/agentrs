"""cam_live.harness.staging_discord — STAGING ONLY.

A local imitation of the subset of the Discord REST API (v10) that ``cam_live.harness.discord`` uses: same paths,
payload shapes, ``Authorization: Bot`` header, rate-limit headers and error codes. It exists so that the adapter's
contract tests and the independent membership verifier can run without any call to Discord.

Nothing obtained from this server is a real-world result: every response carries the header
``X-Staging: cam-live-staging-discord`` and every observation the adapter builds from it is labelled ``staging``.
"""
from .server import (API_PREFIX, STAGING_HEADER, STAGING_LABEL, StagingDiscord, StagingDiscordServer)

__all__ = ["API_PREFIX", "STAGING_HEADER", "STAGING_LABEL", "StagingDiscord", "StagingDiscordServer"]
