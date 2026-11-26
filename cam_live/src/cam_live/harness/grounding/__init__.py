"""cam_live.harness.grounding — replaceable visual-grounding models (workstream EC).

Everything here implements ``contracts.VisualGrounder``: ``ground(image_png, instruction) -> {"x", "y",
"confidence", ...}`` and ``info()``. A grounder proposes a point; it never acts.

=================  =========================================================  ==================  ============
name               model                                                      licence             GPU lock
=================  =========================================================  ==================  ============
goclick-base       HongxinLi/GoClick-Base @ 05633a49 (lightweight default)    MIT                 no (~0.6 GB)
mai-ui-2b          Tongyi-MAI/MAI-UI-2B @ 50305093 (harder interfaces)        Apache-2.0          yes (> 4 GB)
vocaela-2-500m     vocaela/Vocaela-2-500M-1024R2 @ b0280f2c (evaluation only) CC BY-NC-SA 4.0     no
reference-template rendered-text / colour matching — baseline only            (part of cam_live)  no (CPU)
=================  =========================================================  ==================  ============

``load_grounder(name)`` returns a model-backed grounder running in its own process with a clean environment
(:class:`worker.IsolatedGrounder`); ``isolated=False, allow_inprocess=True`` loads it in the calling process, which
is meant for the benchmark and for the worker itself.
"""
from __future__ import annotations

from typing import Any, Dict

from . import goclick, maiui, vocaela
from .base import GrounderLoadRefused, GrounderUnavailable, GroundingError, NeuralGrounder, credential_holder
from .goclick import GoClickGrounder
from .maiui import MaiUIGrounder
from .reference import ReferenceGrounder
from .vocaela import VocaelaGrounder
from .worker import IsolatedGrounder

__all__ = ["REGISTRY", "DEFAULT_ORDER", "load_grounder", "available_grounders", "GoClickGrounder", "MaiUIGrounder",
           "VocaelaGrounder", "ReferenceGrounder", "IsolatedGrounder", "NeuralGrounder", "GroundingError",
           "GrounderUnavailable", "GrounderLoadRefused", "credential_holder"]

_CLASSES = {"goclick-base": GoClickGrounder, "mai-ui-2b": MaiUIGrounder, "vocaela-2-500m": VocaelaGrounder,
            "reference-template": ReferenceGrounder}

REGISTRY: Dict[str, Dict[str, Any]] = {
    "goclick-base": {"repo": goclick.REPO, "revision": goclick.REVISION, "license": "MIT", "role": "lightweight",
                     "needs_gpu_lock": False, "noncommercial": False, "baseline": False},
    "mai-ui-2b": {"repo": maiui.REPO, "revision": maiui.REVISION, "license": "Apache-2.0", "role": "advanced",
                  "needs_gpu_lock": True, "noncommercial": False, "baseline": False},
    "vocaela-2-500m": {"repo": vocaela.REPO, "revision": vocaela.REVISION, "license": "CC BY-NC-SA 4.0",
                       "role": "evaluation only (non-commercial licence)", "needs_gpu_lock": False,
                       "noncommercial": True, "baseline": False},
    "reference-template": {"repo": "", "revision": ReferenceGrounder.revision, "license": "part of cam_live",
                           "role": "baseline for comparison only", "needs_gpu_lock": False, "noncommercial": False,
                           "baseline": True},
}
#: engineering default for trying grounders: lightweight first, the advanced model for what it cannot do
DEFAULT_ORDER = ("goclick-base", "mai-ui-2b")


def load_grounder(name: str, isolated: bool = True, **options: Any):
    """Create a grounder by registry name. Model weights are loaded on first use, not here."""
    if name not in _CLASSES:
        raise KeyError(f"unknown grounder {name!r}; known: {sorted(_CLASSES)}")
    if name == "reference-template":
        options.pop("allow_inprocess", None)
        options.pop("device", None)
        return ReferenceGrounder(**options)
    if isolated:
        options.pop("allow_inprocess", None)
        return IsolatedGrounder(name, **options)
    return _CLASSES[name](**options)


def available_grounders() -> Dict[str, Dict[str, Any]]:
    """Registry entries with ``available`` / ``unavailable_reason`` for this host (no model is loaded)."""
    out: Dict[str, Dict[str, Any]] = {}
    for name, meta in REGISTRY.items():
        entry = dict(meta)
        if meta["baseline"]:
            entry.update(available=True, unavailable_reason="")
        else:
            ok, why = _CLASSES[name](accept_noncommercial=True).available()
            entry.update(available=ok, unavailable_reason=why)
            if ok and meta["noncommercial"]:
                entry["unavailable_reason"] = "usable for evaluation only: non-commercial licence"
        out[name] = entry
    return out
