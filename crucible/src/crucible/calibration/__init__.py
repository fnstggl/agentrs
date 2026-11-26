"""Crucible 5.0 - Empirical calibration: dataset acquisition, behavioural model fitting and held-out evaluation, the fitted-model registry.

Owned by workstream E (docs/v5/WORKSTREAMS.md). Kernel modules of this package live in
`calibration/modules.py` or are imported from it; `crucible.load_all_modules()` imports that file when it exists.

Interface (INTERFACE_CONTRACTS section 5), one registry for every calibrated model and dataset:

    datasets() -> dict[str, dict]       # name -> {url, license, sha256 per file, size, schema, provenance}
    acquire(name, dest=None) -> str     # downloads, verifies checksums, returns the directory
    load(model_name) -> FittedModel     # e.g. "content_consumption@1", "commercial_response@1"; verifies hashes

World code depends only on `load`; it never reads datasets. The downloader (`datasets.acquire`, urllib) is offline
tooling: this package exports `acquire` and `datasets` lazily (module `__getattr__`), so importing the package for
`load` - which world code does - never loads it (checked by the red team's static-reachability probe). Fitting code (`crucible.calibration.fit_e1`,
`fit_commercial`, `fit_software`) is run offline and writes checkpoints, model cards and held-out reports under
`docs/v5/calibration/`. `registry_e2` is a compatibility layer over this interface for E2's fitting scripts.
"""
from .registry import CalibrationError, FittedModel, list_models, load

__all__ = ["CalibrationError", "FittedModel", "acquire", "datasets", "list_models", "load"]


def __getattr__(name):
    if name in ("acquire", "datasets"):
        import importlib
        _ds = importlib.import_module(".datasets", __name__)   # the offline downloader, loaded only when asked for
        fn = getattr(_ds, name)
        globals()[name] = fn                   # keep the function (not the submodule) as the package attribute
        return fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
