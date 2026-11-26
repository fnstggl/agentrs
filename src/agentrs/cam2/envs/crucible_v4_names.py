"""The Crucible 4.0 module names of the relocated CAM adapters, for reproducing Crucible 4.0 tests unchanged.

Crucible 5.0 moved `crucible.integrations.agentrs_cam2` and `crucible.integrations.agentrs_sl` to the AGENTRS side
(`agentrs.cam2.envs.crucible_adapter`, `agentrs.sl.crucible_world`; Crucible DECISION_LOG D-81: Crucible never imports
CAM). Crucible's Gate A.1 runs the Crucible 4.0 regression suite byte-identical to its 4.0 commit (D-82); the 4.0
`tests/test_agents_llm.py` imports the adapters under their 4.0 names. Loading this module (as a pytest plugin,
``-p agentrs.cam2.envs.crucible_v4_names``) maps exactly those two names to the relocated modules, nothing else.
It lives here, on the AGENTRS side, because the mapping points from Crucible names into AGENTRS code.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys

ALIASES = {"crucible.integrations.agentrs_cam2": "agentrs.cam2.envs.crucible_adapter",
           "crucible.integrations.agentrs_sl": "agentrs.sl.crucible_world"}


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name not in ALIASES:
            return None
        return importlib.util.spec_from_loader(name, self)

    def create_module(self, spec):
        return importlib.import_module(ALIASES[spec.name])

    def exec_module(self, module):          # the relocated module is already executed
        return None


def install() -> None:
    if not any(isinstance(f, _Alias) for f in sys.meta_path):
        sys.meta_path.insert(0, _Alias())


install()
