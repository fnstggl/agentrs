"""World-hosting latch (Part 16: the real-world adapter is disabled by default and cannot be enabled where agents act).

Every entry point that hosts synthetic worlds - the CLI, the HTTP/MCP API servers, the web app, the experience
runner and the process-pool workers - calls `engage(<entry>)` before anything else. It sets
``CRUCIBLE_HOSTS_WORLDS`` in the process environment (inherited by every child process) and engages
`crucible.realworld.forbid_in_this_process` when the bridge is already loaded; `RealWorldBridge` refuses to be
constructed in such a process. This module imports nothing from the bridge, so world code may import it.
"""
from __future__ import annotations

import os
import sys

ENV = "CRUCIBLE_HOSTS_WORLDS"


def engage(entry: str) -> None:
    os.environ[ENV] = os.environ.get(ENV) or str(entry)
    br = sys.modules.get("crucible.realworld.bridge")
    if br is not None:
        br.forbid_in_this_process(f"this process hosts synthetic worlds ({entry})")


def engaged() -> str | None:
    return os.environ.get(ENV) or None
