"""Authorisation-gated bridge for real-world validation experiments. OFF BY DEFAULT.

This package is completely separate from synthetic evaluation:

* nothing under `crucible.core`, `crucible.services`, `crucible.economy`, `crucible.actors`, `crucible.worlds`,
  `crucible.sandbox` or `crucible.goals` imports it (checked by `tests/test_realworld_gate.py` and by the red-team
  probe `independence.world_code_cannot_import_evaluator_or_bridge`), so no tool call can reach it and no in-world
  actor can enable it;
* a `RealWorldBridge` constructed without an authorisation document refuses every request;
* an authorisation document is a file written by the human operator (`authorization.load_authorization`); a file
  owned by another user (e.g. the sandbox uid), writable by others, or inside a sandbox directory is refused;
* every outbound request is checked against the document's allow-list after DNS resolution and again on every
  redirect hop, private-network and metadata addresses are refused unless listed explicitly, and credentials in
  URLs are refused;
* consequential (non-GET/HEAD) actions need a live approval callback and fail closed without one;
* every request, response, refusal, approval and measurement is appended to a hash-chained log;
* outcomes are produced only by `RealWorldBridge.measure(name, fn)` with a measurement function supplied by the
  experimenter - there is no API through which an agent's claim becomes a result;
* results are `RealWorldResult` objects that the kernel's encoders, `json`, `pickle` and `copy` all refuse, and
  that can only be exported to a `*.realworld.json` file carrying the origin tag.

See docs/SECURITY.md (operating modes) for how this relates to `crucible.modes.RealWorldGateway` (Mode C), which
only gates; this bridge performs the request.
"""
from .authorization import Authorization, AuthorizationError, load_authorization  # noqa: F401
from .bridge import BridgeRefused, RealResponse, RealWorldBridge, forbid_in_this_process  # noqa: F401
from .log import ChainedLog, LogCorrupt, verify_log  # noqa: F401
from .netguard import ExternalConnectionAttempt, NetworkGuard  # noqa: F401
from .results import (ContaminationError, HoldoutBudgetExceeded, RealWorldResult, SealedHoldoutGuard,  # noqa: F401
                      assert_no_real_world, export_results, is_real_world)
